builtin.module {
  func.func @test_add(%0 : memref<3x3xf32>, %1 : memref<3x3xf32>) -> memref<3x3xf32> {
    %2 = memref.alloc() : memref<3x3xf32>
    %3 = arith.constant 0 : index
    %4 = arith.constant 3 : index
    %5 = arith.constant 3 : index
    %6 = arith.constant 1 : index
    scf.for %7 = %3 to %4 step %6 {
      scf.for %8 = %3 to %5 step %6 {
        %9 = memref.load %0[%7, %8] : memref<3x3xf32>
        %10 = memref.load %1[%7, %8] : memref<3x3xf32>
        %11 = arith.addf %9, %10 : f32
        memref.store %11, %2[%7, %8] : memref<3x3xf32>
      }
    }
    func.return %2 : memref<3x3xf32>
  }
  func.func @main() {
    func.return
  }
}