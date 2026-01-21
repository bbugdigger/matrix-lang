"""
Matrix Language Compiler - Custom Dialect Definition

This module defines the Matrix dialect with custom types and operations
for matrix computations. It uses xDSL's IRDL (IR Definition Language)
to define the operations declaratively.
"""

from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    irdl_attr_definition,
    operand_def,
    result_def,
    attr_def,
    ParametrizedAttribute,
    VarOperand,
    var_operand_def,
)
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    IntAttr,
    ArrayAttr,
    TypeAttribute,
    FloatAttr,
    AnyFloat,
)
from xdsl.ir import Dialect, SSAValue, VerifyException
from xdsl.utils.exceptions import VerifyException
from typing import Sequence, ClassVar


# =============================================================================
# Custom Matrix Type
# =============================================================================

@irdl_attr_definition
class MatrixType(ParametrizedAttribute, TypeAttribute):
    """
    Type for matrix with statically known dimensions.
    
    Syntax: !matrix.type<rows x cols x dtype>
    Example: !matrix.type<3 x 3 x f32>
    
    Parameters:
        rows: Number of rows (must be positive)
        cols: Number of columns (must be positive)
        dtype: Element data type (f32, f64, etc.)
    """
    name = "matrix.type"
    
    rows: IntAttr
    cols: IntAttr
    dtype: AnyFloat
    
    def __init__(self, rows: int, cols: int, dtype):
        """Create a MatrixType with given dimensions and element type."""
        super().__init__(
            IntAttr(rows),
            IntAttr(cols),
            dtype
        )
    
    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape as (rows, cols) tuple."""
        return (self.rows.data, self.cols.data)
    
    def __str__(self) -> str:
        return f"!matrix.type<{self.rows.data}x{self.cols.data}x{self.dtype}>"


# =============================================================================
# Matrix Operations
# =============================================================================

@irdl_op_definition
class AllocOp(IRDLOperation):
    """
    Allocate a new matrix in memory.
    
    Syntax: %result = matrix.alloc : !matrix.type<rows x cols x dtype>
    
    This operation allocates uninitialized memory for a matrix.
    The shape is determined by the result type.
    
    Attributes:
        shape: ArrayAttr containing [rows, cols]
    
    Results:
        result: The allocated matrix
    """
    name = "matrix.alloc"
    
    shape = attr_def(ArrayAttr)
    result = result_def(MatrixType)
    
    def __init__(self, rows: int, cols: int, dtype: TypeAttribute):
        """Create an AllocOp for a matrix with given dimensions."""
        matrix_type = MatrixType(rows, cols, dtype)
        super().__init__(
            operands=[],
            attributes={"shape": ArrayAttr([IntAttr(rows), IntAttr(cols)])},
            result_types=[matrix_type]
        )
    
    def verify_(self) -> None:
        """Verify that shape attribute matches result type."""
        shape_data = self.shape.data
        if len(shape_data) != 2:
            raise VerifyException(f"Shape must have 2 elements, got {len(shape_data)}")
        
        rows = shape_data[0].data
        cols = shape_data[1].data
        result_type = self.result.type
        
        if rows != result_type.rows.data:
            raise VerifyException(
                f"Shape rows ({rows}) doesn't match result type rows ({result_type.rows.data})"
            )
        if cols != result_type.cols.data:
            raise VerifyException(
                f"Shape cols ({cols}) doesn't match result type cols ({result_type.cols.data})"
            )


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """
    Create a matrix with constant values.
    
    Syntax: %result = matrix.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : !matrix.type<2x2xf32>
    
    Attributes:
        value: ArrayAttr containing the matrix values (flattened row-major)
        shape: ArrayAttr containing [rows, cols]
    
    Results:
        result: The constant matrix
    """
    name = "matrix.constant"
    
    value = attr_def(ArrayAttr)
    shape = attr_def(ArrayAttr)
    result = result_def(MatrixType)
    
    def __init__(self, values: list[list[float]], dtype: TypeAttribute):
        """Create a ConstantOp with given values."""
        rows = len(values)
        cols = len(values[0]) if values else 0
        matrix_type = MatrixType(rows, cols, dtype)
        
        # Flatten values to row-major order
        flat_values = []
        for row in values:
            for val in row:
                flat_values.append(FloatAttr(val, dtype))
        
        super().__init__(
            operands=[],
            attributes={
                "value": ArrayAttr(flat_values),
                "shape": ArrayAttr([IntAttr(rows), IntAttr(cols)])
            },
            result_types=[matrix_type]
        )
    
    def verify_(self) -> None:
        """Verify that value count matches shape."""
        shape_data = self.shape.data
        rows = shape_data[0].data
        cols = shape_data[1].data
        expected_count = rows * cols
        actual_count = len(self.value.data)
        
        if actual_count != expected_count:
            raise VerifyException(
                f"Value count ({actual_count}) doesn't match shape ({rows}x{cols}={expected_count})"
            )


@irdl_op_definition
class AddOp(IRDLOperation):
    """
    Element-wise matrix addition.
    
    Syntax: %result = matrix.add %lhs, %rhs : !matrix.type<MxNxdtype>
    
    Both operands must have the same dimensions.
    
    Operands:
        lhs: Left-hand side matrix
        rhs: Right-hand side matrix
    
    Results:
        result: Result matrix (same shape as inputs)
    """
    name = "matrix.add"
    
    lhs = operand_def(MatrixType)
    rhs = operand_def(MatrixType)
    result = result_def(MatrixType)
    
    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        """Create an AddOp for two matrices."""
        # Result has same type as inputs
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type]
        )
    
    def verify_(self) -> None:
        """Verify that matrices have compatible dimensions for addition."""
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type
        
        if not isinstance(lhs_type, MatrixType) or not isinstance(rhs_type, MatrixType):
            raise VerifyException("AddOp operands must be MatrixType")
        
        if lhs_type.rows.data != rhs_type.rows.data:
            raise VerifyException(
                f"Matrix addition requires same number of rows: "
                f"lhs has {lhs_type.rows.data}, rhs has {rhs_type.rows.data}"
            )
        
        if lhs_type.cols.data != rhs_type.cols.data:
            raise VerifyException(
                f"Matrix addition requires same number of columns: "
                f"lhs has {lhs_type.cols.data}, rhs has {rhs_type.cols.data}"
            )


@irdl_op_definition
class SubOp(IRDLOperation):
    """
    Element-wise matrix subtraction.
    
    Syntax: %result = matrix.sub %lhs, %rhs : !matrix.type<MxNxdtype>
    
    Both operands must have the same dimensions.
    """
    name = "matrix.sub"
    
    lhs = operand_def(MatrixType)
    rhs = operand_def(MatrixType)
    result = result_def(MatrixType)
    
    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        """Create a SubOp for two matrices."""
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type]
        )
    
    def verify_(self) -> None:
        """Verify that matrices have compatible dimensions for subtraction."""
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type
        
        if not isinstance(lhs_type, MatrixType) or not isinstance(rhs_type, MatrixType):
            raise VerifyException("SubOp operands must be MatrixType")
        
        if lhs_type.rows.data != rhs_type.rows.data:
            raise VerifyException(
                f"Matrix subtraction requires same number of rows: "
                f"lhs has {lhs_type.rows.data}, rhs has {rhs_type.rows.data}"
            )
        
        if lhs_type.cols.data != rhs_type.cols.data:
            raise VerifyException(
                f"Matrix subtraction requires same number of columns: "
                f"lhs has {lhs_type.cols.data}, rhs has {rhs_type.cols.data}"
            )


@irdl_op_definition
class MatMulOp(IRDLOperation):
    """
    Matrix multiplication operation.
    
    Syntax: %result = matrix.matmul %lhs, %rhs : (!matrix.type<MxKxdtype>, !matrix.type<KxNxdtype>) -> !matrix.type<MxNxdtype>
    
    Computes C = A @ B where:
        - A is MxK matrix
        - B is KxN matrix
        - C is MxN matrix
    
    Operands:
        lhs: Left-hand side matrix (MxK)
        rhs: Right-hand side matrix (KxN)
    
    Results:
        result: Result matrix (MxN)
    """
    name = "matrix.matmul"
    
    lhs = operand_def(MatrixType)
    rhs = operand_def(MatrixType)
    result = result_def(MatrixType)
    
    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        """Create a MatMulOp for two matrices."""
        lhs_type = lhs.type
        rhs_type = rhs.type
        
        # Result shape: (lhs.rows, rhs.cols)
        result_type = MatrixType(
            lhs_type.rows.data,
            rhs_type.cols.data,
            lhs_type.dtype
        )
        
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type]
        )
    
    def verify_(self) -> None:
        """Verify that matrix dimensions are compatible for multiplication."""
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type
        
        if not isinstance(lhs_type, MatrixType) or not isinstance(rhs_type, MatrixType):
            raise VerifyException("MatMulOp operands must be MatrixType")
        
        # Check inner dimensions match: lhs.cols == rhs.rows
        if lhs_type.cols.data != rhs_type.rows.data:
            raise VerifyException(
                f"Incompatible matrix dimensions for multiplication: "
                f"({lhs_type.rows.data}x{lhs_type.cols.data}) @ "
                f"({rhs_type.rows.data}x{rhs_type.cols.data}). "
                f"Inner dimensions must match: {lhs_type.cols.data} != {rhs_type.rows.data}"
            )
        
        # Verify result type
        result_type = self.result.type
        expected_rows = lhs_type.rows.data
        expected_cols = rhs_type.cols.data
        
        if result_type.rows.data != expected_rows:
            raise VerifyException(
                f"Result rows ({result_type.rows.data}) doesn't match expected ({expected_rows})"
            )
        if result_type.cols.data != expected_cols:
            raise VerifyException(
                f"Result cols ({result_type.cols.data}) doesn't match expected ({expected_cols})"
            )


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    Matrix transpose operation.
    
    Syntax: %result = matrix.transpose %input : !matrix.type<MxNxdtype> -> !matrix.type<NxMxdtype>
    
    Swaps rows and columns: result[i,j] = input[j,i]
    
    Operands:
        input: Input matrix (MxN)
    
    Results:
        result: Transposed matrix (NxM)
    """
    name = "matrix.transpose"
    
    input = operand_def(MatrixType)
    result = result_def(MatrixType)
    
    def __init__(self, input_val: SSAValue):
        """Create a TransposeOp for a matrix."""
        input_type = input_val.type
        
        # Transpose swaps rows and columns
        result_type = MatrixType(
            input_type.cols.data,
            input_type.rows.data,
            input_type.dtype
        )
        
        super().__init__(
            operands=[input_val],
            result_types=[result_type]
        )
    
    def verify_(self) -> None:
        """Verify that result dimensions are correctly swapped."""
        input_type = self.input.type
        result_type = self.result.type
        
        if not isinstance(input_type, MatrixType) or not isinstance(result_type, MatrixType):
            raise VerifyException("TransposeOp operand and result must be MatrixType")
        
        if result_type.rows.data != input_type.cols.data:
            raise VerifyException(
                f"Transpose result rows ({result_type.rows.data}) must equal input cols ({input_type.cols.data})"
            )
        if result_type.cols.data != input_type.rows.data:
            raise VerifyException(
                f"Transpose result cols ({result_type.cols.data}) must equal input rows ({input_type.rows.data})"
            )


@irdl_op_definition
class ScalarMulOp(IRDLOperation):
    """
    Element-wise scalar multiplication.
    
    Syntax: %result = matrix.scalar_mul %matrix {scalar = 2.0 : f32} : !matrix.type<MxNxdtype>
    
    Multiplies each element of the matrix by a scalar value.
    
    Operands:
        matrix: Input matrix
    
    Attributes:
        scalar: The scalar multiplier (FloatAttr)
    
    Results:
        result: Result matrix (same shape as input)
    """
    name = "matrix.scalar_mul"
    
    matrix = operand_def(MatrixType)
    scalar = attr_def(FloatAttr)
    result = result_def(MatrixType)
    
    def __init__(self, matrix: SSAValue, scalar: float):
        """Create a ScalarMulOp."""
        matrix_type = matrix.type
        super().__init__(
            operands=[matrix],
            attributes={"scalar": FloatAttr(scalar, matrix_type.dtype)},
            result_types=[matrix_type]
        )
    
    def verify_(self) -> None:
        """Verify that result has same shape as input."""
        input_type = self.matrix.type
        result_type = self.result.type
        
        if not isinstance(input_type, MatrixType) or not isinstance(result_type, MatrixType):
            raise VerifyException("ScalarMulOp operand and result must be MatrixType")
        
        if input_type.rows.data != result_type.rows.data:
            raise VerifyException("ScalarMulOp result must have same rows as input")
        if input_type.cols.data != result_type.cols.data:
            raise VerifyException("ScalarMulOp result must have same cols as input")


@irdl_op_definition
class ElementMulOp(IRDLOperation):
    """
    Element-wise (Hadamard) matrix multiplication.
    
    Syntax: %result = matrix.element_mul %lhs, %rhs : !matrix.type<MxNxdtype>
    
    Computes result[i,j] = lhs[i,j] * rhs[i,j]
    Both operands must have the same dimensions.
    """
    name = "matrix.element_mul"
    
    lhs = operand_def(MatrixType)
    rhs = operand_def(MatrixType)
    result = result_def(MatrixType)
    
    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        """Create an ElementMulOp for two matrices."""
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type]
        )
    
    def verify_(self) -> None:
        """Verify that matrices have compatible dimensions."""
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type
        
        if not isinstance(lhs_type, MatrixType) or not isinstance(rhs_type, MatrixType):
            raise VerifyException("ElementMulOp operands must be MatrixType")
        
        if lhs_type.rows.data != rhs_type.rows.data:
            raise VerifyException(
                f"Element-wise multiplication requires same number of rows: "
                f"lhs has {lhs_type.rows.data}, rhs has {rhs_type.rows.data}"
            )
        
        if lhs_type.cols.data != rhs_type.cols.data:
            raise VerifyException(
                f"Element-wise multiplication requires same number of columns: "
                f"lhs has {lhs_type.cols.data}, rhs has {rhs_type.cols.data}"
            )


@irdl_op_definition 
class PrintOp(IRDLOperation):
    """
    Print a matrix to stdout (for debugging/output).
    
    Syntax: matrix.print %matrix : !matrix.type<MxNxdtype>
    
    This operation has side effects and produces no result.
    """
    name = "matrix.print"
    
    input = operand_def(MatrixType)
    
    def __init__(self, input_val: SSAValue):
        """Create a PrintOp."""
        super().__init__(
            operands=[input_val],
            result_types=[]
        )


# =============================================================================
# Matrix Dialect Registration
# =============================================================================

class MatrixDialect(Dialect):
    """
    Dialect for matrix operations.
    
    This dialect provides high-level matrix operations that can be
    lowered to standard MLIR dialects (memref, scf, arith) and
    eventually to LLVM IR.
    
    Operations:
        - matrix.alloc: Allocate matrix memory
        - matrix.constant: Create constant matrix
        - matrix.add: Element-wise addition
        - matrix.sub: Element-wise subtraction  
        - matrix.matmul: Matrix multiplication
        - matrix.transpose: Matrix transpose
        - matrix.scalar_mul: Scalar multiplication
        - matrix.element_mul: Element-wise multiplication
        - matrix.print: Print matrix
    
    Types:
        - !matrix.type<rows x cols x dtype>: Matrix type with dimensions
    """
    name = "matrix"
    
    def __init__(self):
        super().__init__(
            "matrix",
            [
                # Operations
                AllocOp,
                ConstantOp,
                AddOp,
                SubOp,
                MatMulOp,
                TransposeOp,
                ScalarMulOp,
                ElementMulOp,
                PrintOp,
            ],
            [
                # Types
                MatrixType,
            ]
        )
