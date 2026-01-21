"""
Matrix Language Compiler - Standard Lowering Pass

This module converts Matrix dialect operations to standard MLIR dialects
(memref, scf, arith, func). This prepares the IR for subsequent LLVM
conversion using mlir-opt.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from xdsl.context import Context
from xdsl.ir import Operation, SSAValue, Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier
)
from xdsl.dialects import builtin, func, memref, arith, scf, llvm
from xdsl.dialects.builtin import (
    ModuleOp,
    Float32Type,
    Float64Type,
    IndexType,
    IntegerAttr,
    IntegerType,
    FloatAttr,
    UnrealizedConversionCastOp,
    StringAttr,
    i32,
    i64,
    i8,
)

from .dialect import (
    MatrixType,
    MatMulOp,
    TransposeOp,
    ScalarMulOp,
    AddOp,
    SubOp,
    ElementMulOp,
    AllocOp,
    ConstantOp,
    PrintOp,
)


def convert_matrix_type(matrix_type: MatrixType) -> memref.MemRefType:
    """Convert MatrixType to MemRefType."""
    rows = matrix_type.rows.data
    cols = matrix_type.cols.data
    element_type = matrix_type.dtype
    return memref.MemRefType(element_type, [rows, cols])


def get_operand_as_memref(operand: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    """Get an operand as a memref, inserting a cast if necessary."""
    if isinstance(operand.type, memref.MemRefType):
        return operand
    elif isinstance(operand.type, MatrixType):
        # Insert a cast to convert matrix type to memref
        memref_type = convert_matrix_type(operand.type)
        cast = UnrealizedConversionCastOp.get([operand], [memref_type])
        rewriter.insert_op_before_matched_op(cast)
        return cast.results[0]
    else:
        return operand


@dataclass
class AllocOpLowering(RewritePattern):
    """Lower matrix.alloc to memref.alloc."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AllocOp, rewriter: PatternRewriter):
        # Get shape from attributes
        shape_attr = op.attributes['shape']
        rows = shape_attr.data[0].data
        cols = shape_attr.data[1].data
        
        # Get element type (default to f32)
        element_type = Float32Type()
        
        # Create memref type
        memref_type = memref.MemRefType(element_type, [rows, cols])
        
        # Create memref.alloc operation
        alloc_op = memref.AllocOp([], [], memref_type)
        
        # Cast back to matrix type to maintain interface
        matrix_type = op.result.type
        cast = UnrealizedConversionCastOp.get([alloc_op.results[0]], [matrix_type])
        
        # Replace the operation
        rewriter.replace_matched_op([alloc_op, cast], cast.results)


@dataclass
class ConstantOpLowering(RewritePattern):
    """Lower matrix.constant to memref.alloc + stores."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter):
        # Get shape and values
        shape_attr = op.attributes['shape']
        rows = shape_attr.data[0].data
        cols = shape_attr.data[1].data
        values = op.attributes['value'].data
        
        element_type = Float32Type()
        memref_type = memref.MemRefType(element_type, [rows, cols])
        
        # Allocate memory
        alloc_op = memref.AllocOp([], [], memref_type)
        ops_to_insert = [alloc_op]
        
        # Store each value
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                val = values[idx].value.data
                
                # Create index constants
                i_const = arith.ConstantOp(IntegerAttr(i, IndexType()))
                j_const = arith.ConstantOp(IntegerAttr(j, IndexType()))
                val_const = arith.ConstantOp(FloatAttr(val, element_type))
                
                # Store the value
                store_op = memref.StoreOp.get(val_const, alloc_op, [i_const, j_const])
                
                ops_to_insert.extend([i_const, j_const, val_const, store_op])
        
        # Cast back to matrix type
        matrix_type = op.result.type
        cast = UnrealizedConversionCastOp.get([alloc_op.results[0]], [matrix_type])
        ops_to_insert.append(cast)
        
        rewriter.replace_matched_op(ops_to_insert, cast.results)


@dataclass
class AddOpLowering(RewritePattern):
    """Lower matrix.add to nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter):
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return  # Already lowered
        
        rows = result_type.rows.data
        cols = result_type.cols.data
        element_type = result_type.dtype
        
        # Create result memref
        memref_type = memref.MemRefType(element_type, [rows, cols])
        result = memref.AllocOp([], [], memref_type)
        
        # Get operands as memrefs
        lhs_memref = get_operand_as_memref(op.lhs, rewriter)
        rhs_memref = get_operand_as_memref(op.rhs, rewriter)
        
        # Create constants for loop bounds
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        # Create outer loop
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        # Create inner loop
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        # Inside inner loop: load, add, store
        lhs_elem = memref.LoadOp.get(lhs_memref, [i, j])
        rhs_elem = memref.LoadOp.get(rhs_memref, [i, j])
        sum_elem = arith.AddfOp(lhs_elem, rhs_elem)
        store_op = memref.StoreOp.get(sum_elem, result, [i, j])
        
        # Build inner loop body
        inner_block.add_ops([lhs_elem, rhs_elem, sum_elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        # Add inner loop to outer loop body
        outer_block.add_op(inner_loop)
        outer_block.add_op(scf.YieldOp())
        
        # Cast result back to matrix type
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        rewriter.replace_matched_op(
            [result, zero, rows_bound, cols_bound, one, outer_loop, result_cast],
            result_cast.results
        )


@dataclass
class SubOpLowering(RewritePattern):
    """Lower matrix.sub to nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SubOp, rewriter: PatternRewriter):
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return
        
        rows = result_type.rows.data
        cols = result_type.cols.data
        element_type = result_type.dtype
        
        memref_type = memref.MemRefType(element_type, [rows, cols])
        result = memref.AllocOp([], [], memref_type)
        
        lhs_memref = get_operand_as_memref(op.lhs, rewriter)
        rhs_memref = get_operand_as_memref(op.rhs, rewriter)
        
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        lhs_elem = memref.LoadOp.get(lhs_memref, [i, j])
        rhs_elem = memref.LoadOp.get(rhs_memref, [i, j])
        diff_elem = arith.SubfOp(lhs_elem, rhs_elem)
        store_op = memref.StoreOp.get(diff_elem, result, [i, j])
        
        inner_block.add_ops([lhs_elem, rhs_elem, diff_elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        outer_block.add_op(inner_loop)
        outer_block.add_op(scf.YieldOp())
        
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        rewriter.replace_matched_op(
            [result, zero, rows_bound, cols_bound, one, outer_loop, result_cast],
            result_cast.results
        )


@dataclass
class TransposeOpLowering(RewritePattern):
    """Lower matrix.transpose to nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter):
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return
        
        # For transpose, result has swapped dimensions
        out_rows = result_type.rows.data
        out_cols = result_type.cols.data
        element_type = result_type.dtype
        
        # Input dimensions
        in_rows = out_cols
        in_cols = out_rows
        
        memref_type = memref.MemRefType(element_type, [out_rows, out_cols])
        result = memref.AllocOp([], [], memref_type)
        
        input_memref = get_operand_as_memref(op.input, rewriter)
        
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(in_rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(in_cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        # Load from [i,j], store to [j,i]
        elem = memref.LoadOp.get(input_memref, [i, j])
        store_op = memref.StoreOp.get(elem, result, [j, i])
        
        inner_block.add_ops([elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        outer_block.add_op(inner_loop)
        outer_block.add_op(scf.YieldOp())
        
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        rewriter.replace_matched_op(
            [result, zero, rows_bound, cols_bound, one, outer_loop, result_cast],
            result_cast.results
        )


@dataclass
class ScalarMulOpLowering(RewritePattern):
    """Lower matrix.scalar_mul to nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ScalarMulOp, rewriter: PatternRewriter):
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return
        
        rows = result_type.rows.data
        cols = result_type.cols.data
        element_type = result_type.dtype
        
        scalar_attr = op.attributes['scalar']
        scalar_value = scalar_attr.value.data
        
        memref_type = memref.MemRefType(element_type, [rows, cols])
        result = memref.AllocOp([], [], memref_type)
        
        matrix_memref = get_operand_as_memref(op.matrix, rewriter)
        
        scalar_const = arith.ConstantOp(FloatAttr(scalar_value, element_type))
        
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        elem = memref.LoadOp.get(matrix_memref, [i, j])
        mul_elem = arith.MulfOp(elem, scalar_const)
        store_op = memref.StoreOp.get(mul_elem, result, [i, j])
        
        inner_block.add_ops([elem, mul_elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        outer_block.add_op(inner_loop)
        outer_block.add_op(scf.YieldOp())
        
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        rewriter.replace_matched_op(
            [result, scalar_const, zero, rows_bound, cols_bound, one, outer_loop, result_cast],
            result_cast.results
        )


@dataclass
class ElementMulOpLowering(RewritePattern):
    """Lower matrix.element_mul to nested loops."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ElementMulOp, rewriter: PatternRewriter):
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return
        
        rows = result_type.rows.data
        cols = result_type.cols.data
        element_type = result_type.dtype
        
        memref_type = memref.MemRefType(element_type, [rows, cols])
        result = memref.AllocOp([], [], memref_type)
        
        lhs_memref = get_operand_as_memref(op.lhs, rewriter)
        rhs_memref = get_operand_as_memref(op.rhs, rewriter)
        
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        lhs_elem = memref.LoadOp.get(lhs_memref, [i, j])
        rhs_elem = memref.LoadOp.get(rhs_memref, [i, j])
        mul_elem = arith.MulfOp(lhs_elem, rhs_elem)
        store_op = memref.StoreOp.get(mul_elem, result, [i, j])
        
        inner_block.add_ops([lhs_elem, rhs_elem, mul_elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        outer_block.add_op(inner_loop)
        outer_block.add_op(scf.YieldOp())
        
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        rewriter.replace_matched_op(
            [result, zero, rows_bound, cols_bound, one, outer_loop, result_cast],
            result_cast.results
        )


@dataclass
class PrintOpLowering(RewritePattern):
    """Lower matrix.print to nested loops with calls to external print functions.
    
    This lowering creates:
    1. Nested loops to iterate over matrix elements
    2. Calls to external 'print_f32' function for each element
    3. Calls to external 'print_newline' at the end of each row
    
    The external functions must be provided by a runtime library. A minimal
    runtime can be created with this C code:
    
    ```c
    // matrix_runtime.c
    #include <stdio.h>
    void print_f32(float val) { printf("%8.4f ", val); }
    void print_newline(void) { printf("\\n"); }
    ```
    
    Compile with: clang -c matrix_runtime.c -o matrix_runtime.o
    Link with: clang program.o matrix_runtime.o -o program
    """
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PrintOp, rewriter: PatternRewriter):
        input_type = op.input.type
        if not isinstance(input_type, MatrixType):
            # Already lowered or wrong type - just remove
            rewriter.erase_matched_op()
            return
        
        rows = input_type.rows.data
        cols = input_type.cols.data
        element_type = input_type.dtype
        
        # Get the input as memref
        input_memref = get_operand_as_memref(op.input, rewriter)
        
        # Create index constants for loop bounds
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        # Create outer loop (rows)
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        # Create inner loop (cols)
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        # Load element
        elem = memref.LoadOp.get(input_memref, [i, j])
        inner_block.add_op(elem)
        
        # Call print_f32 external function
        print_f32_ref = builtin.FlatSymbolRefAttr(StringAttr("print_f32"))
        call_print = func.CallOp(print_f32_ref, [elem.results[0]], [])
        inner_block.add_op(call_print)
        
        inner_block.add_op(scf.YieldOp())
        
        # After inner loop, call print_newline
        print_newline_ref = builtin.FlatSymbolRefAttr(StringAttr("print_newline"))
        call_newline = func.CallOp(print_newline_ref, [], [])
        
        # Add inner loop and newline call to outer loop
        outer_block.add_op(inner_loop)
        outer_block.add_op(call_newline)
        outer_block.add_op(scf.YieldOp())
        
        # Replace the print operation with the loops
        rewriter.replace_matched_op(
            [zero, rows_bound, cols_bound, one, outer_loop],
            []
        )


@dataclass
class MatMulOpLowering(RewritePattern):
    """Lower matrix.matmul to triple nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatMulOp, rewriter: PatternRewriter):
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return
        
        m = result_type.rows.data  # rows of result
        n = result_type.cols.data  # cols of result
        
        # Get k from lhs type (cols of lhs)
        lhs_type = op.lhs.type
        if isinstance(lhs_type, MatrixType):
            k = lhs_type.cols.data
        else:
            k = lhs_type.shape.data[1].data
        
        element_type = result_type.dtype
        
        memref_type = memref.MemRefType(element_type, [m, n])
        result = memref.AllocOp([], [], memref_type)
        
        lhs_memref = get_operand_as_memref(op.lhs, rewriter)
        rhs_memref = get_operand_as_memref(op.rhs, rewriter)
        
        # Zero constant for initialization
        zero_float = arith.ConstantOp(FloatAttr(0.0, element_type))
        
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        m_bound = arith.ConstantOp(IntegerAttr(m, IndexType()))
        n_bound = arith.ConstantOp(IntegerAttr(n, IndexType()))
        k_bound = arith.ConstantOp(IntegerAttr(k, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        # Initialize result matrix to zero
        init_outer_block = Block(arg_types=[IndexType()])
        init_outer = scf.ForOp(zero, m_bound, one, [], init_outer_block)
        i_init = init_outer_block.args[0]
        
        init_inner_block = Block(arg_types=[IndexType()])
        init_inner = scf.ForOp(zero, n_bound, one, [], init_inner_block)
        j_init = init_inner_block.args[0]
        
        init_store = memref.StoreOp.get(zero_float, result, [i_init, j_init])
        init_inner_block.add_ops([init_store])
        init_inner_block.add_op(scf.YieldOp())
        
        init_outer_block.add_op(init_inner)
        init_outer_block.add_op(scf.YieldOp())
        
        # Matrix multiplication triple loop
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, m_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        middle_block = Block(arg_types=[IndexType()])
        middle_loop = scf.ForOp(zero, n_bound, one, [], middle_block)
        j = middle_block.args[0]
        
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, k_bound, one, [], inner_block)
        k_idx = inner_block.args[0]
        
        # Load elements
        lhs_elem = memref.LoadOp.get(lhs_memref, [i, k_idx])
        rhs_elem = memref.LoadOp.get(rhs_memref, [k_idx, j])
        result_elem = memref.LoadOp.get(result, [i, j])
        
        # Multiply and accumulate
        mul_elem = arith.MulfOp(lhs_elem, rhs_elem)
        acc_elem = arith.AddfOp(result_elem, mul_elem)
        
        # Store back
        store_op = memref.StoreOp.get(acc_elem, result, [i, j])
        
        inner_block.add_ops([lhs_elem, rhs_elem, result_elem, mul_elem, acc_elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        middle_block.add_op(inner_loop)
        middle_block.add_op(scf.YieldOp())
        
        outer_block.add_op(middle_loop)
        outer_block.add_op(scf.YieldOp())
        
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        rewriter.replace_matched_op(
            [result, zero_float, zero, m_bound, n_bound, k_bound, one,
             init_outer, outer_loop, result_cast],
            result_cast.results
        )


class MatrixToStandardLoweringPass(ModulePass):
    """Pass to lower Matrix dialect operations to standard MLIR dialects."""
    
    name = "matrix-to-standard"
    
    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Apply the lowering pass to the module."""
        
        # Lower matrix operations to memref/scf/arith
        patterns = [
            AllocOpLowering(),
            ConstantOpLowering(),
            AddOpLowering(),
            SubOpLowering(),
            TransposeOpLowering(),
            ScalarMulOpLowering(),
            ElementMulOpLowering(),
            MatMulOpLowering(),
            PrintOpLowering(),
        ]
        
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(patterns),
            apply_recursively=True
        )
        walker.rewrite_module(op)
        
        # Update function signatures
        self._update_functions(op)
        
        # Clean up redundant casts
        self._cleanup_casts(op)
    
    def _update_functions(self, module: ModuleOp):
        """Update function signatures to use memref types."""
        for op in module.body.ops:
            if isinstance(op, func.FuncOp):
                func_type = op.function_type
                new_inputs = []
                new_outputs = []
                
                for input_type in func_type.inputs.data:
                    if isinstance(input_type, MatrixType):
                        new_inputs.append(convert_matrix_type(input_type))
                    else:
                        new_inputs.append(input_type)
                
                for output_type in func_type.outputs.data:
                    if isinstance(output_type, MatrixType):
                        new_outputs.append(convert_matrix_type(output_type))
                    else:
                        new_outputs.append(output_type)
                
                new_func_type = func.FunctionType.from_lists(new_inputs, new_outputs)
                op.attributes['function_type'] = new_func_type
                
                # Update block arguments
                if op.body.blocks:
                    entry_block = op.body.blocks[0]
                    for i, arg in enumerate(entry_block.args):
                        if i < len(new_inputs):
                            arg._type = new_inputs[i]
    
    def _cleanup_casts(self, module: ModuleOp):
        """Remove redundant conversion casts."""
        ops_to_remove = []
        
        for op in module.walk():
            if isinstance(op, UnrealizedConversionCastOp):
                if op.inputs and op.results:
                    if len(op.inputs) == 1 and len(op.results) == 1:
                        input_type = op.inputs[0].type
                        result_type = op.results[0].type
                        
                        # If input is already the right type, bypass
                        if input_type == result_type:
                            for use in list(op.results[0].uses):
                                use.operation.operands[use.index] = op.inputs[0]
                            ops_to_remove.append(op)
        
        for op in ops_to_remove:
            op.detach()


def lower_to_standard_dialects(ctx: Context, module: ModuleOp, verbose: bool = False) -> ModuleOp:
    """Main entry point for lowering Matrix dialect to standard MLIR dialects."""
    lowering_pass = MatrixToStandardLoweringPass()
    lowering_pass.apply(ctx, module)
    
    # Add external function declarations for print support
    _add_print_function_declarations(module)
    
    # Final cleanup to ensure all matrix types are eliminated
    cleanup_final_types(ctx, module)
    
    if verbose:
        print("Successfully lowered Matrix dialect to standard MLIR dialects")
    
    return module


def _add_print_function_declarations(module: ModuleOp):
    """Add external function declarations for print_f32 and print_newline.
    
    These functions must be provided by a runtime library. Example implementation:
    
    ```c
    // matrix_runtime.c
    #include <stdio.h>
    void print_f32(float val) { printf("%8.4f ", val); }
    void print_newline(void) { printf("\\n"); }
    ```
    """
    # Check if we need to add declarations (only if print functions are called)
    needs_print_f32 = False
    needs_print_newline = False
    
    for op in module.walk():
        if isinstance(op, func.CallOp):
            callee_name = op.callee.string_value()
            if callee_name == "print_f32":
                needs_print_f32 = True
            elif callee_name == "print_newline":
                needs_print_newline = True
    
    if not needs_print_f32 and not needs_print_newline:
        return  # No print functions used
    
    # Get the module's body block
    body_block = module.body.blocks[0]
    
    # Find first operation to insert before
    first_op = None
    for op in body_block.ops:
        first_op = op
        break
    
    # Add print_f32 declaration: func.func private @print_f32(f32)
    if needs_print_f32:
        print_f32_type = func.FunctionType.from_lists([Float32Type()], [])
        print_f32_decl = func.FuncOp("print_f32", print_f32_type, Region(), visibility="private")
        if first_op:
            body_block.insert_op_before(print_f32_decl, first_op)
        else:
            body_block.add_op(print_f32_decl)
        first_op = print_f32_decl
    
    # Add print_newline declaration: func.func private @print_newline()
    if needs_print_newline:
        print_newline_type = func.FunctionType.from_lists([], [])
        print_newline_decl = func.FuncOp("print_newline", print_newline_type, Region(), visibility="private")
        if first_op:
            body_block.insert_op_after(print_newline_decl, first_op)
        else:
            body_block.add_op(print_newline_decl)


def cleanup_final_types(ctx: Context, module: ModuleOp) -> ModuleOp:
    """Final cleanup to ensure all matrix types are eliminated."""
    
    # First, remove all UnrealizedConversionCastOps that involve matrix types
    def remove_matrix_casts(op: Operation):
        """Recursively remove casts involving matrix types."""
        ops_to_remove = []
        
        # Process all operations in the current operation's regions
        for region in op.regions:
            for block in region.blocks:
                for block_op in list(block.ops):
                    if isinstance(block_op, UnrealizedConversionCastOp):
                        # Check if this cast involves matrix types or is redundant
                        has_matrix = False
                        is_redundant = False
                        
                        for result in block_op.results:
                            if isinstance(result.type, MatrixType):
                                has_matrix = True
                                break
                        
                        # Check if it's a redundant cast (memref to same memref)
                        if block_op.inputs and block_op.results:
                            if (len(block_op.inputs) == 1 and len(block_op.results) == 1 and
                                isinstance(block_op.inputs[0].type, memref.MemRefType) and
                                isinstance(block_op.results[0].type, memref.MemRefType) and
                                block_op.inputs[0].type == block_op.results[0].type):
                                is_redundant = True
                        
                        if (has_matrix or is_redundant) and block_op.inputs:
                            # Replace uses of the cast result with the input
                            for i, result in enumerate(block_op.results):
                                if i < len(block_op.inputs):
                                    for use in list(result.uses):
                                        use.operation.operands[use.index] = block_op.inputs[i]
                            ops_to_remove.append(block_op)
                    else:
                        # Recursively process nested operations
                        remove_matrix_casts(block_op)
        
        # Remove collected operations
        for op_to_remove in ops_to_remove:
            op_to_remove.detach()
    
    # Remove matrix casts from the entire module
    remove_matrix_casts(module)
    
    # Update function signatures using proper xDSL API
    for op in module.body.ops:
        if isinstance(op, func.FuncOp):
            # Skip external function declarations
            if op.is_declaration:
                continue
                
            # First, update block argument types from MatrixType to memref
            if op.body.blocks:
                entry_block = op.body.blocks[0]
                for arg in entry_block.args:
                    if isinstance(arg.type, MatrixType):
                        # Directly update the block argument type
                        arg._type = convert_matrix_type(arg.type)
            
            # Update function calls within this function
            for block in op.body.blocks:
                for block_op in list(block.ops):
                    if isinstance(block_op, func.CallOp):
                        # Check if any result type needs updating
                        needs_update = False
                        new_result_types = []
                        for result_type in block_op.result_types:
                            if isinstance(result_type, MatrixType):
                                new_result_types.append(convert_matrix_type(result_type))
                                needs_update = True
                            else:
                                new_result_types.append(result_type)
                        
                        if needs_update:
                            # Create a new CallOp with updated types
                            new_call = func.CallOp(
                                block_op.callee,
                                list(block_op.operands),
                                new_result_types
                            )
                            # Replace the old op
                            block.insert_op_before(new_call, block_op)
                            # Update uses of the old result
                            for old_res, new_res in zip(block_op.results, new_call.results):
                                for use in list(old_res.uses):
                                    use.operation.operands[use.index] = new_res
                            block_op.detach()
            
            # Use xDSL's built-in method to update function type
            # This reads block argument types and return operand types automatically
            op.update_function_type()
    
    return module
