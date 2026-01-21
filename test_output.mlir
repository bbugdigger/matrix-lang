builtin.module {
  func.func @matrix_computation(%0 : memref<3x3xf32>, %1 : memref<3x3xf32>) -> memref<3x3xf32> {
    %2 = memref.alloc() : memref<3x3xf32>
    %3 = arith.constant 0.000000e+00 : f32
    %4 = arith.constant 0 : index
    %5 = arith.constant 3 : index
    %6 = arith.constant 3 : index
    %7 = arith.constant 3 : index
    %8 = arith.constant 1 : index
    scf.for %9 = %4 to %5 step %8 {
      scf.for %10 = %4 to %6 step %8 {
        memref.store %3, %2[%9, %10] : memref<3x3xf32>
      }
    }
    scf.for %11 = %4 to %5 step %8 {
      scf.for %12 = %4 to %6 step %8 {
        scf.for %13 = %4 to %7 step %8 {
          %14 = memref.load %0[%11, %13] : memref<3x3xf32>
          %15 = memref.load %1[%13, %12] : memref<3x3xf32>
          %16 = memref.load %2[%11, %12] : memref<3x3xf32>
          %17 = arith.mulf %14, %15 : f32
          %18 = arith.addf %16, %17 : f32
          memref.store %18, %2[%11, %12] : memref<3x3xf32>
        }
      }
    }
    %19 = memref.alloc() : memref<3x3xf32>
    %20 = arith.constant 0 : index
    %21 = arith.constant 3 : index
    %22 = arith.constant 3 : index
    %23 = arith.constant 1 : index
    scf.for %24 = %20 to %21 step %23 {
      scf.for %25 = %20 to %22 step %23 {
        %26 = memref.load %2[%24, %25] : memref<3x3xf32>
        %27 = memref.load %1[%24, %25] : memref<3x3xf32>
        %28 = arith.addf %26, %27 : f32
        memref.store %28, %19[%24, %25] : memref<3x3xf32>
      }
    }
    func.return %19 : memref<3x3xf32>
  }
  func.func @simple_matmul(%0 : memref<3x3xf32>, %1 : memref<3x3xf32>) -> memref<3x3xf32> {
    %2 = memref.alloc() : memref<3x3xf32>
    %3 = arith.constant 0.000000e+00 : f32
    %4 = arith.constant 0 : index
    %5 = arith.constant 3 : index
    %6 = arith.constant 3 : index
    %7 = arith.constant 3 : index
    %8 = arith.constant 1 : index
    scf.for %9 = %4 to %5 step %8 {
      scf.for %10 = %4 to %6 step %8 {
        memref.store %3, %2[%9, %10] : memref<3x3xf32>
      }
    }
    scf.for %11 = %4 to %5 step %8 {
      scf.for %12 = %4 to %6 step %8 {
        scf.for %13 = %4 to %7 step %8 {
          %14 = memref.load %0[%11, %13] : memref<3x3xf32>
          %15 = memref.load %1[%13, %12] : memref<3x3xf32>
          %16 = memref.load %2[%11, %12] : memref<3x3xf32>
          %17 = arith.mulf %14, %15 : f32
          %18 = arith.addf %16, %17 : f32
          memref.store %18, %2[%11, %12] : memref<3x3xf32>
        }
      }
    }
    func.return %2 : memref<3x3xf32>
  }
  func.func @transpose_chain(%0 : memref<3x3xf32>) -> memref<3x3xf32> {
    %1 = memref.alloc() : memref<3x3xf32>
    %2 = arith.constant 0 : index
    %3 = arith.constant 3 : index
    %4 = arith.constant 3 : index
    %5 = arith.constant 1 : index
    scf.for %6 = %2 to %3 step %5 {
      scf.for %7 = %2 to %4 step %5 {
        %8 = memref.load %0[%6, %7] : memref<3x3xf32>
        memref.store %8, %1[%7, %6] : memref<3x3xf32>
      }
    }
    func.return %1 : memref<3x3xf32>
  }
  func.func @scalar_operations(%0 : memref<3x3xf32>) -> memref<3x3xf32> {
    %1 = memref.alloc() : memref<3x3xf32>
    %2 = arith.constant 2.000000e+00 : f32
    %3 = arith.constant 0 : index
    %4 = arith.constant 3 : index
    %5 = arith.constant 3 : index
    %6 = arith.constant 1 : index
    scf.for %7 = %3 to %4 step %6 {
      scf.for %8 = %3 to %5 step %6 {
        %9 = memref.load %0[%7, %8] : memref<3x3xf32>
        %10 = arith.mulf %9, %2 : f32
        memref.store %10, %1[%7, %8] : memref<3x3xf32>
      }
    }
    %11 = memref.alloc() : memref<3x3xf32>
    %12 = arith.constant 5.000000e-01 : f32
    %13 = arith.constant 0 : index
    %14 = arith.constant 3 : index
    %15 = arith.constant 3 : index
    %16 = arith.constant 1 : index
    scf.for %17 = %13 to %14 step %16 {
      scf.for %18 = %13 to %15 step %16 {
        %19 = memref.load %1[%17, %18] : memref<3x3xf32>
        %20 = arith.mulf %19, %12 : f32
        memref.store %20, %11[%17, %18] : memref<3x3xf32>
      }
    }
    func.return %11 : memref<3x3xf32>
  }
  func.func @main() {
    %0 = memref.alloc() : memref<3x3xf32>
    %1 = arith.constant 0 : index
    %2 = arith.constant 0 : index
    %3 = arith.constant 1.000000e+00 : f32
    memref.store %3, %0[%1, %2] : memref<3x3xf32>
    %4 = arith.constant 0 : index
    %5 = arith.constant 1 : index
    %6 = arith.constant 2.000000e+00 : f32
    memref.store %6, %0[%4, %5] : memref<3x3xf32>
    %7 = arith.constant 0 : index
    %8 = arith.constant 2 : index
    %9 = arith.constant 3.000000e+00 : f32
    memref.store %9, %0[%7, %8] : memref<3x3xf32>
    %10 = arith.constant 1 : index
    %11 = arith.constant 0 : index
    %12 = arith.constant 4.000000e+00 : f32
    memref.store %12, %0[%10, %11] : memref<3x3xf32>
    %13 = arith.constant 1 : index
    %14 = arith.constant 1 : index
    %15 = arith.constant 5.000000e+00 : f32
    memref.store %15, %0[%13, %14] : memref<3x3xf32>
    %16 = arith.constant 1 : index
    %17 = arith.constant 2 : index
    %18 = arith.constant 6.000000e+00 : f32
    memref.store %18, %0[%16, %17] : memref<3x3xf32>
    %19 = arith.constant 2 : index
    %20 = arith.constant 0 : index
    %21 = arith.constant 7.000000e+00 : f32
    memref.store %21, %0[%19, %20] : memref<3x3xf32>
    %22 = arith.constant 2 : index
    %23 = arith.constant 1 : index
    %24 = arith.constant 8.000000e+00 : f32
    memref.store %24, %0[%22, %23] : memref<3x3xf32>
    %25 = arith.constant 2 : index
    %26 = arith.constant 2 : index
    %27 = arith.constant 9.000000e+00 : f32
    memref.store %27, %0[%25, %26] : memref<3x3xf32>
    %28 = memref.alloc() : memref<3x3xf32>
    %29 = arith.constant 0 : index
    %30 = arith.constant 0 : index
    %31 = arith.constant 9.000000e+00 : f32
    memref.store %31, %28[%29, %30] : memref<3x3xf32>
    %32 = arith.constant 0 : index
    %33 = arith.constant 1 : index
    %34 = arith.constant 8.000000e+00 : f32
    memref.store %34, %28[%32, %33] : memref<3x3xf32>
    %35 = arith.constant 0 : index
    %36 = arith.constant 2 : index
    %37 = arith.constant 7.000000e+00 : f32
    memref.store %37, %28[%35, %36] : memref<3x3xf32>
    %38 = arith.constant 1 : index
    %39 = arith.constant 0 : index
    %40 = arith.constant 6.000000e+00 : f32
    memref.store %40, %28[%38, %39] : memref<3x3xf32>
    %41 = arith.constant 1 : index
    %42 = arith.constant 1 : index
    %43 = arith.constant 5.000000e+00 : f32
    memref.store %43, %28[%41, %42] : memref<3x3xf32>
    %44 = arith.constant 1 : index
    %45 = arith.constant 2 : index
    %46 = arith.constant 4.000000e+00 : f32
    memref.store %46, %28[%44, %45] : memref<3x3xf32>
    %47 = arith.constant 2 : index
    %48 = arith.constant 0 : index
    %49 = arith.constant 3.000000e+00 : f32
    memref.store %49, %28[%47, %48] : memref<3x3xf32>
    %50 = arith.constant 2 : index
    %51 = arith.constant 1 : index
    %52 = arith.constant 2.000000e+00 : f32
    memref.store %52, %28[%50, %51] : memref<3x3xf32>
    %53 = arith.constant 2 : index
    %54 = arith.constant 2 : index
    %55 = arith.constant 1.000000e+00 : f32
    memref.store %55, %28[%53, %54] : memref<3x3xf32>
    %56 = func.call @matrix_computation(%0, %28) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
    func.return
  }
}