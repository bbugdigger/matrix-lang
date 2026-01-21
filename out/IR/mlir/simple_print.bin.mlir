builtin.module {
  func.func private @print_f32(f32) -> () 
  func.func private @print_newline() -> () 
  func.func @main() {
    %0 = memref.alloc() : memref<2x2xf32>
    %1 = arith.constant 0 : index
    %2 = arith.constant 0 : index
    %3 = arith.constant 1.000000e+00 : f32
    memref.store %3, %0[%1, %2] : memref<2x2xf32>
    %4 = arith.constant 0 : index
    %5 = arith.constant 1 : index
    %6 = arith.constant 2.000000e+00 : f32
    memref.store %6, %0[%4, %5] : memref<2x2xf32>
    %7 = arith.constant 1 : index
    %8 = arith.constant 0 : index
    %9 = arith.constant 3.000000e+00 : f32
    memref.store %9, %0[%7, %8] : memref<2x2xf32>
    %10 = arith.constant 1 : index
    %11 = arith.constant 1 : index
    %12 = arith.constant 4.000000e+00 : f32
    memref.store %12, %0[%10, %11] : memref<2x2xf32>
    %13 = arith.constant 0 : index
    %14 = arith.constant 2 : index
    %15 = arith.constant 2 : index
    %16 = arith.constant 1 : index
    scf.for %17 = %13 to %14 step %16 {
      scf.for %18 = %13 to %15 step %16 {
        %19 = memref.load %0[%17, %18] : memref<2x2xf32>
        func.call @print_f32(%19) : (f32) -> ()
      }
      func.call @print_newline() : () -> ()
    }
    func.return
  }
}