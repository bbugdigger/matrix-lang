"""
Matrix Language Compiler Package

A compiler for matrix-oriented programming language using xDSL framework.
"""

from .dialect import (
    MatrixType,
    MatrixDialect,
    AllocOp,
    ConstantOp,
    AddOp,
    SubOp,
    MatMulOp,
    TransposeOp,
    ScalarMulOp,
    ElementMulOp,
    PrintOp,
)

from .parser import parse_matrix_program, MatrixOperationExtractor

from .ir_generator import MatrixIRGenerator

from .middle_end import (
    DoubleTransposeElimination,
    DeadCodeElimination,
    ConstantFolding,
    MatrixOptimizationPipeline,
)

from .standard_lowering import (
    MatrixToStandardLoweringPass,
    lower_to_standard_dialects,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "MatrixType",
    "MatrixDialect",
    # Operations
    "AllocOp",
    "ConstantOp", 
    "AddOp",
    "SubOp",
    "MatMulOp",
    "TransposeOp",
    "ScalarMulOp",
    "ElementMulOp",
    "PrintOp",
    # Parser
    "parse_matrix_program",
    "MatrixOperationExtractor",
    # IR Generator
    "MatrixIRGenerator",
    # Optimizations
    "DoubleTransposeElimination",
    "DeadCodeElimination",
    "ConstantFolding",
    "MatrixOptimizationPipeline",
    # Lowering
    "MatrixToStandardLoweringPass",
    "lower_to_standard_dialects",
]
