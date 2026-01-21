"""
Matrix Language Compiler - Parser

This module parses Python-like source code with matrix operations
and extracts operation information for IR generation.
"""

import ast
from typing import Dict, List, Any, Optional


class MatrixOperationExtractor(ast.NodeVisitor):
    """Extract matrix operations from Python AST."""
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.current_function: Optional[str] = None
        self.variables: Dict[str, Dict[str, Any]] = {}
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Process function definition."""
        self.current_function = node.name
        self.functions[node.name] = {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "operations": [],
            "docstring": ast.get_docstring(node)
        }
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
        
        self.current_function = None
    
    def visit_Assign(self, node: ast.Assign):
        """Process assignment statements."""
        if not self.current_function:
            return
        
        target = node.targets[0]
        if isinstance(target, ast.Name):
            var_name = target.id
            
            # Analyze the right-hand side
            op_info = self.analyze_expression(node.value)
            
            operation = {
                "type": "assign",
                "target": var_name,
                "operation": op_info
            }
            
            self.functions[self.current_function]["operations"].append(operation)
            
            self.variables[var_name] = {
                "type": "matrix",
                "defined_by": op_info
            }
    
    def visit_Return(self, node: ast.Return):
        """Process return statement."""
        if not self.current_function or not node.value:
            return
        
        op_info = self.analyze_expression(node.value)
        operation = {
            "type": "return",
            "operation": op_info
        }
        self.functions[self.current_function]["operations"].append(operation)
    
    def visit_Expr(self, node: ast.Expr):
        """Process expression statements (like print calls)."""
        if not self.current_function:
            return
        
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id
                if func_name == "print":
                    args = [self.analyze_expression(arg) for arg in node.value.args]
                    operation = {
                        "type": "print",
                        "args": args
                    }
                    self.functions[self.current_function]["operations"].append(operation)
    
    def analyze_expression(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze expression to identify matrix operations."""
        
        # Matrix multiplication (A @ B)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return {
                "op": "matmul",
                "left": self.analyze_expression(node.left),
                "right": self.analyze_expression(node.right)
            }
        
        # Element-wise multiplication (A * B or A * scalar)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = self.analyze_expression(node.left)
            right = self.analyze_expression(node.right)
            
            # Check if one is a scalar
            if right.get("op") == "constant":
                return {
                    "op": "scalar_mul",
                    "matrix": left,
                    "scalar": right["value"]
                }
            elif left.get("op") == "constant":
                return {
                    "op": "scalar_mul",
                    "matrix": right,
                    "scalar": left["value"]
                }
            else:
                return {
                    "op": "element_mul",
                    "left": left,
                    "right": right
                }
        
        # Matrix addition (A + B)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return {
                "op": "add",
                "left": self.analyze_expression(node.left),
                "right": self.analyze_expression(node.right)
            }
        
        # Matrix subtraction (A - B)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            return {
                "op": "sub",
                "left": self.analyze_expression(node.left),
                "right": self.analyze_expression(node.right)
            }
        
        # Transpose (A.T)
        elif isinstance(node, ast.Attribute) and node.attr == "T":
            inner = self.analyze_expression(node.value)
            return {
                "op": "transpose",
                "matrix": inner
            }
        
        # Variable reference
        elif isinstance(node, ast.Name):
            return {
                "op": "var",
                "name": node.id
            }
        
        # Constant value
        elif isinstance(node, ast.Constant):
            return {
                "op": "constant",
                "value": node.value
            }
        
        # For older Python versions
        elif isinstance(node, ast.Num):
            return {
                "op": "constant",
                "value": node.n
            }
        
        # Matrix allocation (matrix([[1, 2], [3, 4]]))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                
                if func_name == "matrix":
                    # Extract matrix literal values
                    if node.args and isinstance(node.args[0], ast.List):
                        rows = []
                        for row in node.args[0].elts:
                            if isinstance(row, ast.List):
                                row_values = [self.get_value(elem) for elem in row.elts]
                                rows.append(row_values)
                        
                        return {
                            "op": "alloc",
                            "shape": [len(rows), len(rows[0]) if rows else 0],
                            "values": rows
                        }
                
                elif func_name == "zeros":
                    if len(node.args) >= 2:
                        rows = self.get_value(node.args[0])
                        cols = self.get_value(node.args[1])
                        return {
                            "op": "zeros",
                            "shape": [rows, cols]
                        }
                
                elif func_name == "ones":
                    if len(node.args) >= 2:
                        rows = self.get_value(node.args[0])
                        cols = self.get_value(node.args[1])
                        return {
                            "op": "ones",
                            "shape": [rows, cols]
                        }
                
                elif func_name == "identity" or func_name == "eye":
                    if len(node.args) >= 1:
                        n = self.get_value(node.args[0])
                        return {
                            "op": "identity",
                            "size": n
                        }
                
                else:
                    # Function call
                    args = [self.analyze_expression(arg) for arg in node.args]
                    return {
                        "op": "call",
                        "function": func_name,
                        "args": args
                    }
        
        return {"op": "unknown", "node": ast.dump(node)}
    
    def get_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.get_value(node.operand)
        return None


def parse_matrix_program(source_code: str) -> Dict[str, Any]:
    """Parse source code and extract matrix operations by function."""
    tree = ast.parse(source_code)
    extractor = MatrixOperationExtractor()
    extractor.visit(tree)
    return extractor.functions
