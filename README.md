# Matrix Language Compiler

A compiler for a matrix-oriented domain-specific language (DSL) built using the [xDSL](https://github.com/xdslproject/xdsl) framework. It compiles high-level matrix operations written in Python-like syntax down to native executables through MLIR and LLVM.

## Compilation Pipeline

```
  Source Code (.mx)
        │
        ▼
  ┌─────────────────┐
  │  Python Parser  │  Parse Python AST
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  IR Generator   │  Build Matrix dialect operations
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │   Middle-End    │  Optimization passes:
  │   Optimizer     │  - DoubleTransposeElimination
  │                 │  - TransposeAddFusion
  │                 │  - ConstantFolding
  │                 │  - Dead Code Elimination
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │    Lowering     │  Lower to standard MLIR dialects
  │                 │  (memref, scf, arith, func)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │   LLVM Tools    │  mlir-opt → mlir-translate → llc → clang
  └────────┬────────┘
           │
           ▼
     Executable
```

## Project Structure

```
matrix-lang/
├── compiler.py              # Main compiler entry point
├── compile_full.sh          # Full compilation pipeline script
├── pyproject.toml           # Python package configuration
├── src/
│   ├── __init__.py          # Package exports
│   ├── dialect.py           # Matrix dialect definition (operations, types)
│   ├── parser.py            # Python AST-based parser
│   ├── ir_generator.py      # IR generation with SSA management
│   ├── middle_end.py        # Optimization passes
│   └── standard_lowering.py # Lowering to standard MLIR dialects
├── runtime/
│   └── matrix_runtime.c     # C runtime library (print functions)
├── examples/                # Example .mx programs
│   ├── example.mx
│   ├── simple_matmul.mx
│   ├── transpose_test.mx
│   └── ...
└── out/                     # Build output (generated)
    ├── IR/
    │   ├── mlir/            # .mlir files
    │   └── llvm/            # .ll and .s files
    └── *.bin                # Final executables
```

## Output Structure

When compiling a program, intermediate files are organized as:

```
out/
├── IR/
│   ├── mlir/
│   │   ├── <name>.mlir        # Standard MLIR (memref/scf/arith)
│   │   └── <name>_llvm.mlir   # LLVM dialect MLIR
│   └── llvm/
│       ├── <name>.ll          # LLVM IR
│       └── <name>.s           # Assembly
└── <name>.bin                 # Final executable
```

## Language Features

```python
def matrix_computation(A, B):
    C = A @ B          # Matrix multiplication
    D = A + B          # Element-wise addition
    E = A - B          # Element-wise subtraction
    F = A * 2.0        # Scalar multiplication
    G = A * B          # Element-wise multiplication
    H = A.T            # Transpose
    print(H)           # Print matrix
    return H
```

## Requirements

**Python:**
- Python >= 3.10
- xDSL >= 0.14.0

**LLVM Toolchain (for full compilation):**
- `mlir-opt` and `mlir-translate` (MLIR tools)
- `llc` (LLVM static compiler)
- `clang` (C compiler for linking)

On Ubuntu/Debian:
```bash
# Install LLVM 18 toolchain
sudo apt install mlir-18-tools llvm-18 clang-18

# Install Python dependencies
pip install xdsl
```

## Usage

**Compile a Matrix program:**
```bash
./compile_full.sh examples/example.mx my_program
```

**Run the compiled program:**
```bash
./out/my_program.bin
```

**Generate only MLIR (no LLVM tools required):**
```bash
python compiler.py examples/example.mx --emit-standard -o output.mlir
```

**View optimization passes:**
```bash
python compiler.py examples/example.mx --verbose
```

## Example

Input (`examples/simple_matmul.mx`):
```python
def matmul(A, B):
    return A @ B
```

Generated MLIR (simplified):
```mlir
func.func @matmul(%A: memref<3x3xf32>, %B: memref<3x3xf32>) -> memref<3x3xf32> {
  %result = memref.alloc() : memref<3x3xf32>
  // Nested loops for matrix multiplication
  scf.for %i = %c0 to %c3 step %c1 {
    scf.for %j = %c0 to %c3 step %c1 {
      scf.for %k = %c0 to %c3 step %c1 {
        // result[i,j] += A[i,k] * B[k,j]
      }
    }
  }
  return %result : memref<3x3xf32>
}
```
