#!/usr/bin/env python3
"""
Matrix Operations Compiler

A complete compiler for matrix operations with optimization.
Compiles matrix DSL code through custom IR to LLVM IR.
"""

import sys
import argparse
from pathlib import Path

from src.parser import parse_matrix_program
from src.dialect import MatrixDialect
from src.ir_generator import MatrixIRGenerator
from src.middle_end import MatrixOptimizationPipeline
from src.standard_lowering import lower_to_standard_dialects

from xdsl.context import Context
from xdsl.printer import Printer
from xdsl.dialects import builtin, func, llvm, memref, arith, scf


class MatrixCompiler:
    """Main compiler class for the Matrix language."""
    
    def __init__(self):
        """Initialize the compiler with xDSL context."""
        # Initialize xDSL context
        self.ctx = Context()
        self.ctx.load_dialect(builtin.Builtin)
        self.ctx.load_dialect(func.Func)
        self.ctx.load_dialect(MatrixDialect())
        self.ctx.load_dialect(memref.MemRef)
        self.ctx.load_dialect(arith.Arith)
        self.ctx.load_dialect(scf.Scf)
        self.ctx.load_dialect(llvm.LLVM)
        
        self.printer = Printer()
    
    def compile_file(self, input_file: Path,
                    output_file: Path = None,
                    optimize: bool = True,
                    emit_standard: bool = False,
                    verbose: bool = False) -> builtin.ModuleOp:
        """
        Compile a source file with matrix operations.
        
        Args:
            input_file: Path to input source file
            output_file: Path to output file (optional)
            optimize: Whether to run optimizations
            emit_standard: Whether to lower to standard MLIR dialects
            verbose: Whether to print verbose output
            
        Returns:
            The compiled ModuleOp
        """
        if verbose:
            print(f"Compiling {input_file}...")
        
        # Step 1: Parse source
        with open(input_file, 'r') as f:
            source_code = f.read()
        
        functions = parse_matrix_program(source_code)
        if verbose:
            total_ops = sum(len(f['operations']) for f in functions.values())
            print(f"Parsed {len(functions)} functions with {total_ops} operations")
        
        # Step 2: Generate IR
        generator = MatrixIRGenerator()
        module = generator.generate(functions)
        if verbose:
            print("Generated Matrix IR")
        
        # Step 3: Optimize (if enabled)
        if optimize:
            if verbose:
                print("Running optimization passes...")
            optimizer = MatrixOptimizationPipeline()
            optimizer.apply(self.ctx, module)
        else:
            if verbose:
                print("Skipping optimizations")
        
        # Step 4: Lower to standard dialects (if requested)
        if emit_standard:
            if verbose:
                print("Lowering to standard MLIR dialects...")
            module = lower_to_standard_dialects(self.ctx, module, verbose=verbose)
        
        # Step 5: Output IR
        if output_file:
            with open(output_file, 'w') as f:
                printer = Printer(stream=f)
                printer.print_op(module)
            if verbose:
                dialect_type = "standard MLIR" if emit_standard else "Matrix IR"
                print(f"Wrote {dialect_type} to {output_file}")
        else:
            print("\nGenerated IR:")
            self.printer.print_op(module)
        
        return module
    
    def compile_string(self, source_code: str,
                      optimize: bool = True,
                      emit_standard: bool = False) -> builtin.ModuleOp:
        """
        Compile source code string.
        
        Args:
            source_code: Source code string
            optimize: Whether to run optimizations
            emit_standard: Whether to lower to standard MLIR dialects
            
        Returns:
            The compiled ModuleOp
        """
        functions = parse_matrix_program(source_code)
        
        generator = MatrixIRGenerator()
        module = generator.generate(functions)
        
        if optimize:
            optimizer = MatrixOptimizationPipeline()
            optimizer.apply(self.ctx, module)
        
        if emit_standard:
            module = lower_to_standard_dialects(self.ctx, module)
        
        return module


def main():
    """Main entry point for the compiler CLI."""
    parser = argparse.ArgumentParser(
        description="Compile Matrix language to optimized IR"
    )
    parser.add_argument(
        'input',
        help='Input source file with matrix operations'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for generated IR'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Disable optimizations'
    )
    parser.add_argument(
        '--emit-standard',
        action='store_true',
        help='Lower to standard MLIR dialects (prepared for LLVM conversion)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Create compiler
    compiler = MatrixCompiler()
    
    # Compile the file
    try:
        module = compiler.compile_file(
            Path(args.input),
            Path(args.output) if args.output else None,
            optimize=not args.no_optimize,
            emit_standard=args.emit_standard,
            verbose=args.verbose
        )
        
        if args.verbose:
            print("\nCompilation successful!")
            
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
