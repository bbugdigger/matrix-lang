"""
Matrix Language Compiler - Middle End (Optimization Passes)

This module implements optimization passes for the Matrix dialect,
including double transpose elimination and dead code elimination.
"""

from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
)
from xdsl.ir import Operation
from xdsl.dialects import builtin, func
from xdsl.context import Context
from typing import List, Optional

from .dialect import TransposeOp, AddOp, MatMulOp


class DoubleTransposeElimination(RewritePattern):
    """
    Eliminate double transpose patterns: (A^T)^T = A
    
    This optimization recognizes when a transpose operation is applied
    to the result of another transpose operation and eliminates both,
    replacing with the original matrix.
    
    Example:
        %1 = matrix.transpose %0   // A^T
        %2 = matrix.transpose %1   // (A^T)^T = A
        
    Becomes:
        (uses of %2 replaced with %0)
    """
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter) -> None:
        """Match and eliminate double transpose."""
        
        # Check if input to this transpose is also a transpose
        input_val = op.operands[0]
        if not input_val.owner:
            return
        
        input_op = input_val.owner
        
        if isinstance(input_op, TransposeOp):
            # Found pattern: transpose(transpose(X)) = X
            original_matrix = input_op.operands[0]
            
            # Replace all uses of the double transpose result with the original matrix
            op.results[0].replace_by(original_matrix)
            
            # Check if the inner transpose has no other uses
            if not list(input_val.uses):
                # Erase the inner transpose first if it's dead
                rewriter.erase_op(input_op)
            
            # Erase the outer transpose operation
            rewriter.erase_op(op)


class TransposeAddFusion(RewritePattern):
    """
    Fuse transpose with addition when both operands are transposed.
    
    Pattern: (A^T + B^T) = (A + B)^T
    
    This can reduce the number of transpose operations.
    """
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter) -> None:
        """Match and fuse transpose with addition."""
        
        lhs = op.lhs
        rhs = op.rhs
        
        # Check if both operands are transposes
        if not lhs.owner or not rhs.owner:
            return
        
        if not isinstance(lhs.owner, TransposeOp) or not isinstance(rhs.owner, TransposeOp):
            return
        
        # Found pattern: transpose(A) + transpose(B)
        # Transform to: transpose(A + B)
        
        lhs_transpose = lhs.owner
        rhs_transpose = rhs.owner
        
        original_lhs = lhs_transpose.operands[0]
        original_rhs = rhs_transpose.operands[0]
        
        # Create addition of original matrices
        new_add = AddOp(original_lhs, original_rhs)
        rewriter.insert_op_before_matched_op(new_add)
        
        # Create single transpose of the result
        new_transpose = TransposeOp(new_add.result)
        rewriter.insert_op_before_matched_op(new_transpose)
        
        # Replace uses
        op.results[0].replace_by(new_transpose.result)
        
        # Erase the original add
        rewriter.erase_op(op)
        
        # Clean up dead transposes
        if not list(lhs.uses):
            rewriter.erase_op(lhs_transpose)
        if not list(rhs.uses):
            rewriter.erase_op(rhs_transpose)


class DeadCodeElimination(ModulePass):
    """
    Remove operations whose results are never used.
    
    This pass iteratively removes dead operations until no more
    can be eliminated. An operation is dead if:
    - It produces results
    - None of its results are used
    - It has no side effects
    """
    
    name = "dead-code-elimination"
    
    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:
        """Remove dead operations from the module."""
        eliminated = 0
        changed = True
        
        # Operations with side effects that should not be removed
        side_effect_ops = {
            "func.return", 
            "func.call", 
            "matrix.print",
            "memref.store",
        }
        
        # Repeat until no more dead code can be eliminated
        while changed:
            changed = False
            ops_to_remove = []
            
            # Walk through all operations
            for op in module.walk():
                # Skip certain operations that shouldn't be removed
                if isinstance(op, (builtin.ModuleOp, func.FuncOp)):
                    continue
                
                # Skip operations with side effects
                if op.name in side_effect_ops:
                    continue
                
                # Check if any result of this operation is used
                has_uses = False
                for result in op.results:
                    if list(result.uses):
                        has_uses = True
                        break
                
                # If no results are used, mark for removal
                if not has_uses and len(op.results) > 0:
                    ops_to_remove.append(op)
            
            # Remove dead operations
            for op in ops_to_remove:
                op.detach()
                op.erase()
                eliminated += 1
                changed = True
        
        if eliminated > 0:
            print(f"DCE: Eliminated {eliminated} dead operations")


class ConstantFolding(ModulePass):
    """
    Fold constant operations at compile time.
    
    This pass evaluates operations with constant inputs
    at compile time and replaces them with constant results.
    
    Supported foldings:
    - Transpose of constant matrix
    - Addition of two constant matrices
    - Scalar multiplication of constant matrix
    - Matrix multiplication of two constant matrices (small matrices only)
    """
    
    name = "constant-folding"
    
    # Maximum matrix size for constant folding (to avoid compile-time blowup)
    MAX_FOLD_SIZE = 16
    
    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:
        """Apply constant folding to the module."""
        from .dialect import ConstantOp, ScalarMulOp, ElementMulOp
        
        changed = True
        folded_count = 0
        
        while changed:
            changed = False
            ops_to_process = []
            
            # Collect operations that might be foldable
            for op in module.walk():
                if isinstance(op, TransposeOp):
                    ops_to_process.append(('transpose', op))
                elif isinstance(op, AddOp):
                    ops_to_process.append(('add', op))
                elif isinstance(op, ScalarMulOp):
                    ops_to_process.append(('scalar_mul', op))
                elif isinstance(op, MatMulOp):
                    ops_to_process.append(('matmul', op))
                elif isinstance(op, ElementMulOp):
                    ops_to_process.append(('element_mul', op))
            
            for op_type, op in ops_to_process:
                result = self._try_fold(op_type, op)
                if result is not None:
                    # Create a new constant operation with the folded result
                    new_const = ConstantOp(result, op.result.type.dtype)
                    
                    # Insert before the operation
                    op.parent.insert_op_before(new_const, op)
                    
                    # Replace uses
                    op.result.replace_by(new_const.result)
                    
                    # Remove the old operation
                    op.detach()
                    op.erase()
                    
                    changed = True
                    folded_count += 1
                    break  # Restart iteration after modification
        
        if folded_count > 0:
            print(f"ConstantFolding: Folded {folded_count} operations")
    
    def _try_fold(self, op_type: str, op: Operation) -> Optional[List[List[float]]]:
        """Try to fold an operation. Returns the result matrix values or None if not foldable."""
        from .dialect import ConstantOp, ScalarMulOp, ElementMulOp
        
        if op_type == 'transpose':
            return self._fold_transpose(op)
        elif op_type == 'add':
            return self._fold_add(op)
        elif op_type == 'scalar_mul':
            return self._fold_scalar_mul(op)
        elif op_type == 'matmul':
            return self._fold_matmul(op)
        elif op_type == 'element_mul':
            return self._fold_element_mul(op)
        return None
    
    def _get_constant_values(self, val: Operation) -> Optional[List[List[float]]]:
        """Extract constant values from a ConstantOp, returns 2D list or None."""
        from .dialect import ConstantOp
        
        if not val.owner or not isinstance(val.owner, ConstantOp):
            return None
        
        const_op = val.owner
        shape = const_op.shape.data
        rows = shape[0].data
        cols = shape[1].data
        
        # Check size limit
        if rows * cols > self.MAX_FOLD_SIZE * self.MAX_FOLD_SIZE:
            return None
        
        flat_values = [v.value.data for v in const_op.value.data]
        
        # Reshape to 2D
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(flat_values[i * cols + j])
            result.append(row)
        
        return result
    
    def _fold_transpose(self, op: TransposeOp) -> Optional[List[List[float]]]:
        """Fold transpose of a constant matrix."""
        input_vals = self._get_constant_values(op.input)
        if input_vals is None:
            return None
        
        rows = len(input_vals)
        cols = len(input_vals[0]) if input_vals else 0
        
        # Transpose: result[j][i] = input[i][j]
        result = [[input_vals[i][j] for i in range(rows)] for j in range(cols)]
        return result
    
    def _fold_add(self, op: AddOp) -> Optional[List[List[float]]]:
        """Fold addition of two constant matrices."""
        lhs_vals = self._get_constant_values(op.lhs)
        rhs_vals = self._get_constant_values(op.rhs)
        
        if lhs_vals is None or rhs_vals is None:
            return None
        
        rows = len(lhs_vals)
        cols = len(lhs_vals[0]) if lhs_vals else 0
        
        result = [[lhs_vals[i][j] + rhs_vals[i][j] for j in range(cols)] for i in range(rows)]
        return result
    
    def _fold_scalar_mul(self, op) -> Optional[List[List[float]]]:
        """Fold scalar multiplication of a constant matrix."""
        matrix_vals = self._get_constant_values(op.matrix)
        if matrix_vals is None:
            return None
        
        scalar = op.scalar.value.data
        rows = len(matrix_vals)
        cols = len(matrix_vals[0]) if matrix_vals else 0
        
        result = [[matrix_vals[i][j] * scalar for j in range(cols)] for i in range(rows)]
        return result
    
    def _fold_element_mul(self, op) -> Optional[List[List[float]]]:
        """Fold element-wise multiplication of two constant matrices."""
        lhs_vals = self._get_constant_values(op.lhs)
        rhs_vals = self._get_constant_values(op.rhs)
        
        if lhs_vals is None or rhs_vals is None:
            return None
        
        rows = len(lhs_vals)
        cols = len(lhs_vals[0]) if lhs_vals else 0
        
        result = [[lhs_vals[i][j] * rhs_vals[i][j] for j in range(cols)] for i in range(rows)]
        return result
    
    def _fold_matmul(self, op: MatMulOp) -> Optional[List[List[float]]]:
        """Fold matrix multiplication of two constant matrices."""
        lhs_vals = self._get_constant_values(op.lhs)
        rhs_vals = self._get_constant_values(op.rhs)
        
        if lhs_vals is None or rhs_vals is None:
            return None
        
        m = len(lhs_vals)
        k = len(lhs_vals[0]) if lhs_vals else 0
        n = len(rhs_vals[0]) if rhs_vals else 0
        
        # Standard matrix multiplication
        result = []
        for i in range(m):
            row = []
            for j in range(n):
                val = 0.0
                for p in range(k):
                    val += lhs_vals[i][p] * rhs_vals[p][j]
                row.append(val)
            result.append(row)
        
        return result


class MatrixOptimizationPipeline(ModulePass):
    """
    Complete optimization pipeline for matrix operations.
    
    This pipeline applies multiple optimization passes in order:
    1. Constant folding (evaluate constant expressions at compile time)
    2. Double transpose elimination
    3. Transpose-add fusion
    4. Dead code elimination
    """
    
    name = "matrix-opt-pipeline"
    
    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:
        """Apply all matrix optimizations."""
        
        # First, apply constant folding to evaluate constant expressions
        const_fold = ConstantFolding()
        const_fold.apply(ctx, module)
        
        # Apply algebraic simplifications
        patterns = [
            DoubleTransposeElimination(),
            TransposeAddFusion(),
        ]
        
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(patterns),
            apply_recursively=True
        )
        walker.rewrite_module(module)
        
        # Apply dead code elimination
        dce = DeadCodeElimination()
        dce.apply(ctx, module)
        
        # Verify IR is still valid
        module.verify()
