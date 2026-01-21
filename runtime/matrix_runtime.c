/**
 * Matrix Language Runtime Library
 * 
 * This file provides runtime support functions for the Matrix language compiler.
 * Compile with: clang -c matrix_runtime.c -o matrix_runtime.o
 * Link with your compiled program: clang program.o matrix_runtime.o -o program
 */

#include <stdio.h>

/**
 * Print a single-precision floating point value.
 * Format: 8 characters wide, 4 decimal places, followed by space.
 */
void print_f32(float val) {
    printf("%8.4f ", val);
}

/**
 * Print a newline character.
 * Called at the end of each matrix row.
 */
void print_newline(void) {
    printf("\n");
}

/**
 * Print a double-precision floating point value.
 * Format: 8 characters wide, 4 decimal places, followed by space.
 */
void print_f64(double val) {
    printf("%8.4f ", val);
}

/**
 * Print an integer value.
 * Format: 8 characters wide, followed by space.
 */
void print_i32(int val) {
    printf("%8d ", val);
}

/**
 * Print a separator line for matrix output.
 */
void print_separator(void) {
    printf("--------\n");
}

/**
 * Print a label string.
 */
void print_label(const char* label) {
    printf("%s:\n", label);
}
