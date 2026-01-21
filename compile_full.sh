#!/bin/bash

# Complete compilation pipeline for Matrix language
# Usage: ./compile_full.sh <input.mx> [output_name]
#
# This script compiles Matrix language source through the following stages:
# 1. Matrix Source (.mx) -> Matrix IR (custom dialect)
# 2. Matrix IR -> Standard MLIR (memref/scf/arith)
# 3. Standard MLIR -> LLVM Dialect
# 4. LLVM Dialect -> LLVM IR (.ll)
# 5. LLVM IR -> Assembly (.s)
# 6. Assembly + Runtime -> Executable (.bin)
#
# Output structure:
#   out/
#   ├── IR/
#   │   ├── mlir/          # MLIR and custom dialect IR files
#   │   │   ├── <name>.mlir        (Standard MLIR)
#   │   │   └── <name>_llvm.mlir   (LLVM dialect MLIR)
#   │   └── llvm/          # LLVM IR files
#   │       ├── <name>.ll          (LLVM IR)
#   │       └── <name>.s           (Assembly)
#   └── <name>.bin         # Final executable

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.mx> [output_name]"
    echo "Example: $0 examples/example.mx my_program"
    exit 1
fi

INPUT=$1
OUTPUT_NAME=${2:-"output"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/runtime"

# Output directories
OUT_DIR="$SCRIPT_DIR/out"
MLIR_DIR="$OUT_DIR/IR/mlir"
LLVM_DIR="$OUT_DIR/IR/llvm"

# Create output directories
mkdir -p "$MLIR_DIR"
mkdir -p "$LLVM_DIR"

# Find MLIR tools
if command -v mlir-opt-18 &> /dev/null; then
    MLIR_OPT="mlir-opt-18"
    MLIR_TRANSLATE="mlir-translate-18"
elif command -v mlir-opt &> /dev/null; then
    MLIR_OPT="mlir-opt"
    MLIR_TRANSLATE="mlir-translate"
else
    echo "Warning: MLIR tools not found. Will only generate MLIR output."
    MLIR_OPT=""
fi

# Find LLVM tools
if command -v llc-18 &> /dev/null; then
    LLC="llc-18"
    CLANG="clang-18"
elif command -v llc &> /dev/null; then
    LLC="llc"
    CLANG="clang"
else
    LLC=""
    CLANG=""
fi

echo "=== Matrix Language Compiler ==="
echo "Input: $INPUT"
echo "Output: $OUTPUT_NAME"
echo "Output directory: $OUT_DIR"
echo ""

# Step 1: Compile Matrix source to standard MLIR
echo "[1/6] Compiling Matrix source to MLIR..."
MLIR_OUTPUT="$MLIR_DIR/${OUTPUT_NAME}.mlir"
python3 "$SCRIPT_DIR/compiler.py" "$INPUT" --emit-standard -o "$MLIR_OUTPUT" --verbose

if [ ! -f "$MLIR_OUTPUT" ]; then
    echo "Error: Failed to generate MLIR" >&2
    exit 1
fi

echo "  Generated: $MLIR_OUTPUT"

# Step 2: Convert to LLVM dialect (if mlir-opt available)
if [ -n "$MLIR_OPT" ]; then
    echo "[2/6] Converting to LLVM dialect..."
    LLVM_MLIR_OUTPUT="$MLIR_DIR/${OUTPUT_NAME}_llvm.mlir"
    $MLIR_OPT "$MLIR_OUTPUT" \
        -convert-scf-to-cf \
        -convert-cf-to-llvm \
        -convert-func-to-llvm \
        -convert-arith-to-llvm \
        -convert-index-to-llvm \
        -finalize-memref-to-llvm \
        -reconcile-unrealized-casts \
        -o "$LLVM_MLIR_OUTPUT" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to convert to LLVM dialect"
        echo "  The standard MLIR output is still available: $MLIR_OUTPUT"
    else
        echo "  Generated: $LLVM_MLIR_OUTPUT"
        
        # Step 3: Translate to LLVM IR
        echo "[3/6] Translating to LLVM IR..."
        LLVM_IR_OUTPUT="$LLVM_DIR/${OUTPUT_NAME}.ll"
        $MLIR_TRANSLATE --mlir-to-llvmir "$LLVM_MLIR_OUTPUT" -o "$LLVM_IR_OUTPUT" 2>/dev/null
        
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to translate to LLVM IR"
        else
            echo "  Generated: $LLVM_IR_OUTPUT"
            
            # Step 4: Compile to assembly (if llc available)
            if [ -n "$LLC" ]; then
                echo "[4/6] Compiling to assembly..."
                ASM_OUTPUT="$LLVM_DIR/${OUTPUT_NAME}.s"
                $LLC "$LLVM_IR_OUTPUT" -o "$ASM_OUTPUT" 2>/dev/null
                
                if [ $? -ne 0 ]; then
                    echo "Warning: Failed to generate assembly"
                else
                    echo "  Generated: $ASM_OUTPUT"
                    
                    # Step 5: Compile runtime library (if needed and clang available)
                    if [ -n "$CLANG" ]; then
                        RUNTIME_OBJ="$RUNTIME_DIR/matrix_runtime.o"
                        if [ -f "$RUNTIME_DIR/matrix_runtime.c" ]; then
                            echo "[5/6] Compiling runtime library..."
                            $CLANG -c "$RUNTIME_DIR/matrix_runtime.c" -o "$RUNTIME_OBJ" 2>/dev/null
                            if [ $? -eq 0 ]; then
                                echo "  Generated: $RUNTIME_OBJ"
                            else
                                echo "  Warning: Failed to compile runtime library"
                                RUNTIME_OBJ=""
                            fi
                        else
                            echo "[5/6] Runtime library not found, skipping..."
                            RUNTIME_OBJ=""
                        fi
                        
                        # Step 6: Link to executable
                        echo "[6/6] Linking executable..."
                        BIN_OUTPUT="$OUT_DIR/${OUTPUT_NAME}.bin"
                        if [ -n "$RUNTIME_OBJ" ] && [ -f "$RUNTIME_OBJ" ]; then
                            $CLANG "$ASM_OUTPUT" "$RUNTIME_OBJ" -o "$BIN_OUTPUT" 2>/dev/null
                        else
                            $CLANG "$ASM_OUTPUT" -o "$BIN_OUTPUT" 2>/dev/null
                        fi
                        
                        if [ $? -ne 0 ]; then
                            echo "  Warning: Failed to link executable"
                            echo "  If using print functions, ensure runtime library is compiled:"
                            echo "    clang -c $RUNTIME_DIR/matrix_runtime.c -o $RUNTIME_OBJ"
                        else
                            echo "  Generated: $BIN_OUTPUT"
                        fi
                    fi
                fi
            fi
        fi
    fi
else
    echo "[2-6] Skipped (MLIR tools not found)"
fi

echo ""
echo "=== Compilation Summary ==="
echo "Output directory: $OUT_DIR"
echo ""
echo "Generated files:"
[ -f "$MLIR_DIR/${OUTPUT_NAME}.mlir" ] && echo "  - out/IR/mlir/${OUTPUT_NAME}.mlir (Standard MLIR)"
[ -f "$MLIR_DIR/${OUTPUT_NAME}_llvm.mlir" ] && echo "  - out/IR/mlir/${OUTPUT_NAME}_llvm.mlir (LLVM Dialect MLIR)"
[ -f "$LLVM_DIR/${OUTPUT_NAME}.ll" ] && echo "  - out/IR/llvm/${OUTPUT_NAME}.ll (LLVM IR)"
[ -f "$LLVM_DIR/${OUTPUT_NAME}.s" ] && echo "  - out/IR/llvm/${OUTPUT_NAME}.s (Assembly)"
[ -f "$OUT_DIR/${OUTPUT_NAME}.bin" ] && echo "  - out/${OUTPUT_NAME}.bin (Executable)"
echo ""
if [ -f "$OUT_DIR/${OUTPUT_NAME}.bin" ]; then
    echo "Run with: ./out/${OUTPUT_NAME}.bin"
fi
echo "Done!"
