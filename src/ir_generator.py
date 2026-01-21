"""
Matrix Language Compiler - IR Generator

This module converts parsed operation information into xDSL IR
using the Matrix dialect.
"""

from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region, SSAValue
from xdsl.dialects.builtin import Float32Type, ModuleOp, FlatSymbolRefAttr, StringAttr
from typing import Dict, List, Any, Optional

from .dialect import (
    MatrixType,
    MatMulOp,
    TransposeOp,
    AddOp,
    SubOp,
    ScalarMulOp,
    ElementMulOp,
    AllocOp,
    ConstantOp,
    PrintOp,
)


class MatrixIRGenerator:
    """Generate Matrix dialect IR from parsed operations."""
    
    def __init__(self, default_dtype=None):
        """Initialize the IR generator.
        
        Args:
            default_dtype: Default element type for matrices (defaults to f32)
        """
        self.default_dtype = default_dtype or Float32Type()
        self.var_maps: Dict[str, Dict[str, SSAValue]] = {}
        self.matrix_shapes: Dict[str, tuple] = {}  # Track shapes for type inference
    
    def generate(self, functions: Dict[str, Any]) -> ModuleOp:
        """Generate IR module from parsed functions.
        
        Args:
            functions: Dictionary of parsed function information
            
        Returns:
            Complete ModuleOp containing all functions
        """
        func_ops = []
        
        # First pass: collect function signatures for forward references
        self._collect_function_info(functions)
        
        # Second pass: generate each function
        for func_name, func_info in functions.items():
            func_op = self.generate_function(func_name, func_info)
            if func_op:
                func_ops.append(func_op)
        
        # Create and return module
        module = ModuleOp(func_ops)
        return module
    
    def _collect_function_info(self, functions: Dict[str, Any]):
        """Collect information about functions for type inference."""
        for func_name, func_info in functions.items():
            # Analyze operations to infer matrix shapes
            for op in func_info.get("operations", []):
                if op["type"] == "assign":
                    target = op["target"]
                    op_info = op["operation"]
                    shape = self._infer_shape(op_info, func_info)
                    if shape:
                        self.matrix_shapes[f"{func_name}.{target}"] = shape
    
    def _infer_shape(self, op_info: Dict[str, Any], func_info: Dict[str, Any]) -> Optional[tuple]:
        """Infer the shape of an operation result."""
        op_type = op_info.get("op")
        
        if op_type == "alloc":
            shape = op_info.get("shape", [3, 3])
            return tuple(shape)
        
        elif op_type == "zeros" or op_type == "ones":
            shape = op_info.get("shape", [3, 3])
            return tuple(shape)
        
        elif op_type == "identity":
            n = op_info.get("size", 3)
            return (n, n)
        
        return None
    
    def generate_function(self, func_name: str, func_info: Dict[str, Any]) -> Optional[func.FuncOp]:
        """Generate a single function.
        
        Args:
            func_name: Name of the function
            func_info: Parsed function information
            
        Returns:
            FuncOp for the function
        """
        # Create a new variable map for this function
        self.var_maps[func_name] = {}
        var_map = self.var_maps[func_name]
        
        # Determine function signature
        args = func_info.get("args", [])
        operations = func_info.get("operations", [])
        
        # Infer default matrix size from allocations in main
        default_size = self._get_default_matrix_size(operations)
        
        # Create block with appropriate arguments
        if func_name == "main":
            # No arguments for main
            block = Block(arg_types=[])
            input_types = []
            output_types = []
        else:
            # Assume matrix arguments with inferred or default size
            matrix_type = MatrixType(default_size[0], default_size[1], self.default_dtype)
            arg_types = [matrix_type for _ in args]
            block = Block(arg_types=arg_types)
            
            # Map arguments to variables
            for i, arg_name in enumerate(args):
                var_map[arg_name] = block.args[i]
            
            input_types = arg_types
            output_types = [matrix_type] if self._has_return(operations) else []
        
        # Generate operations
        last_result = None
        return_value = None
        
        for op_info in operations:
            if op_info["type"] == "assign":
                result = self.generate_operation(op_info, block, var_map)
                if result:
                    last_result = result
            elif op_info["type"] == "return":
                return_op = op_info["operation"]
                if return_op["op"] == "var":
                    return_value = var_map.get(return_op["name"])
                else:
                    # Generate the return expression
                    return_value = self._generate_expression(return_op, block, var_map)
                if return_value is None:
                    return_value = last_result
            elif op_info["type"] == "print":
                self._generate_print(op_info, block, var_map)
        
        # Add return operation
        if return_value:
            ret_op = func.ReturnOp(return_value)
        else:
            ret_op = func.ReturnOp()
        block.add_op(ret_op)
        
        # Create function region with the block
        region = Region([block])
        
        # Create function type
        func_type = func.FunctionType.from_lists(
            input_types,
            output_types if return_value else []
        )
        
        # Create function operation
        func_op = func.FuncOp(func_name, func_type, region)
        
        return func_op
    
    def _get_default_matrix_size(self, operations: List[Dict]) -> tuple:
        """Get default matrix size from allocations."""
        for op in operations:
            if op["type"] == "assign":
                op_info = op["operation"]
                if op_info.get("op") == "alloc":
                    shape = op_info.get("shape", [3, 3])
                    return tuple(shape)
        return (3, 3)  # Default fallback
    
    def _has_return(self, operations: List[Dict]) -> bool:
        """Check if function has a return statement."""
        return any(op["type"] == "return" for op in operations)
    
    def generate_operation(self, op_info: Dict[str, Any], block: Block, 
                          var_map: Dict[str, SSAValue]) -> Optional[SSAValue]:
        """Generate IR for a single operation.
        
        Args:
            op_info: Operation information from parser
            block: Block to add operations to
            var_map: Mapping from variable names to SSA values
            
        Returns:
            Result SSA value if operation produces one
        """
        target = op_info["target"]
        result = self._generate_expression(op_info["operation"], block, var_map)
        
        # Store result in variable map
        if result:
            var_map[target] = result
        
        return result
    
    def _generate_expression(self, expr: Dict[str, Any], block: Block,
                            var_map: Dict[str, SSAValue]) -> Optional[SSAValue]:
        """Generate IR for an expression.
        
        Args:
            expr: Expression information from parser
            block: Block to add operations to
            var_map: Mapping from variable names to SSA values
            
        Returns:
            Result SSA value
        """
        op_type = expr.get("op")
        
        if op_type == "matmul":
            left = self._generate_expression(expr["left"], block, var_map)
            right = self._generate_expression(expr["right"], block, var_map)
            if left and right:
                matmul_op = MatMulOp(left, right)
                block.add_op(matmul_op)
                return matmul_op.result
        
        elif op_type == "transpose":
            matrix = self._generate_expression(expr["matrix"], block, var_map)
            if matrix:
                transpose_op = TransposeOp(matrix)
                block.add_op(transpose_op)
                return transpose_op.result
        
        elif op_type == "add":
            left = self._generate_expression(expr["left"], block, var_map)
            right = self._generate_expression(expr["right"], block, var_map)
            if left and right:
                add_op = AddOp(left, right)
                block.add_op(add_op)
                return add_op.result
        
        elif op_type == "sub":
            left = self._generate_expression(expr["left"], block, var_map)
            right = self._generate_expression(expr["right"], block, var_map)
            if left and right:
                sub_op = SubOp(left, right)
                block.add_op(sub_op)
                return sub_op.result
        
        elif op_type == "scalar_mul":
            matrix = self._generate_expression(expr["matrix"], block, var_map)
            scalar = expr.get("scalar", 1.0)
            if matrix:
                scalar_mul_op = ScalarMulOp(matrix, scalar)
                block.add_op(scalar_mul_op)
                return scalar_mul_op.result
        
        elif op_type == "element_mul":
            left = self._generate_expression(expr["left"], block, var_map)
            right = self._generate_expression(expr["right"], block, var_map)
            if left and right:
                element_mul_op = ElementMulOp(left, right)
                block.add_op(element_mul_op)
                return element_mul_op.result
        
        elif op_type == "var":
            name = expr["name"]
            return var_map.get(name)
        
        elif op_type == "alloc":
            shape = expr.get("shape", [3, 3])
            values = expr.get("values")
            rows, cols = shape[0], shape[1]
            
            if values:
                # Create constant matrix with values
                const_op = ConstantOp(values, self.default_dtype)
                block.add_op(const_op)
                return const_op.result
            else:
                # Create uninitialized allocation
                alloc_op = AllocOp(rows, cols, self.default_dtype)
                block.add_op(alloc_op)
                return alloc_op.result
        
        elif op_type == "zeros":
            shape = expr.get("shape", [3, 3])
            rows, cols = shape[0], shape[1]
            values = [[0.0] * cols for _ in range(rows)]
            const_op = ConstantOp(values, self.default_dtype)
            block.add_op(const_op)
            return const_op.result
        
        elif op_type == "ones":
            shape = expr.get("shape", [3, 3])
            rows, cols = shape[0], shape[1]
            values = [[1.0] * cols for _ in range(rows)]
            const_op = ConstantOp(values, self.default_dtype)
            block.add_op(const_op)
            return const_op.result
        
        elif op_type == "identity":
            n = expr.get("size", 3)
            values = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            const_op = ConstantOp(values, self.default_dtype)
            block.add_op(const_op)
            return const_op.result
        
        elif op_type == "call":
            func_name = expr["function"]
            args = expr.get("args", [])
            
            # Generate arguments
            arg_vals = []
            for arg in args:
                val = self._generate_expression(arg, block, var_map)
                if val:
                    arg_vals.append(val)
            
            if arg_vals:
                # Create function call
                callee = FlatSymbolRefAttr(StringAttr(func_name))
                
                # Result type is same as first argument (for matrix functions)
                result_type = arg_vals[0].type if arg_vals else None
                
                if result_type:
                    call_op = func.CallOp(callee, arg_vals, [result_type])
                else:
                    call_op = func.CallOp(callee, arg_vals, [])
                
                block.add_op(call_op)
                
                if call_op.results:
                    return call_op.results[0]
        
        return None
    
    def _generate_print(self, op_info: Dict[str, Any], block: Block,
                       var_map: Dict[str, SSAValue]):
        """Generate print operation."""
        for arg in op_info.get("args", []):
            val = self._generate_expression(arg, block, var_map)
            if val:
                print_op = PrintOp(val)
                block.add_op(print_op)
