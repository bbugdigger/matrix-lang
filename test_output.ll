; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @matrix_computation(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %1, 1
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %7, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %8, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %9, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %10, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %12, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %11, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %13, 4, 1
  %29 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9) to i64))
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %29, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %29, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 0, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 3, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 3, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 3, 4, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 1, 4, 1
  br label %37

37:                                               ; preds = %49, %14
  %38 = phi i64 [ %50, %49 ], [ 0, %14 ]
  %39 = icmp slt i64 %38, 3
  br i1 %39, label %40, label %51

40:                                               ; preds = %37
  br label %41

41:                                               ; preds = %44, %40
  %42 = phi i64 [ %48, %44 ], [ 0, %40 ]
  %43 = icmp slt i64 %42, 3
  br i1 %43, label %44, label %49

44:                                               ; preds = %41
  %45 = mul i64 %38, 3
  %46 = add i64 %45, %42
  %47 = getelementptr float, ptr %29, i64 %46
  store float 0.000000e+00, ptr %47, align 4
  %48 = add i64 %42, 1
  br label %41

49:                                               ; preds = %41
  %50 = add i64 %38, 1
  br label %37

51:                                               ; preds = %37
  br label %52

52:                                               ; preds = %84, %51
  %53 = phi i64 [ %85, %84 ], [ 0, %51 ]
  %54 = icmp slt i64 %53, 3
  br i1 %54, label %55, label %86

55:                                               ; preds = %52
  br label %56

56:                                               ; preds = %82, %55
  %57 = phi i64 [ %83, %82 ], [ 0, %55 ]
  %58 = icmp slt i64 %57, 3
  br i1 %58, label %59, label %84

59:                                               ; preds = %56
  br label %60

60:                                               ; preds = %63, %59
  %61 = phi i64 [ %81, %63 ], [ 0, %59 ]
  %62 = icmp slt i64 %61, 3
  br i1 %62, label %63, label %82

63:                                               ; preds = %60
  %64 = mul i64 %53, 3
  %65 = add i64 %64, %61
  %66 = getelementptr float, ptr %1, i64 %65
  %67 = load float, ptr %66, align 4
  %68 = mul i64 %61, 3
  %69 = add i64 %68, %57
  %70 = getelementptr float, ptr %8, i64 %69
  %71 = load float, ptr %70, align 4
  %72 = mul i64 %53, 3
  %73 = add i64 %72, %57
  %74 = getelementptr float, ptr %29, i64 %73
  %75 = load float, ptr %74, align 4
  %76 = fmul float %67, %71
  %77 = fadd float %75, %76
  %78 = mul i64 %53, 3
  %79 = add i64 %78, %57
  %80 = getelementptr float, ptr %29, i64 %79
  store float %77, ptr %80, align 4
  %81 = add i64 %61, 1
  br label %60

82:                                               ; preds = %60
  %83 = add i64 %57, 1
  br label %56

84:                                               ; preds = %56
  %85 = add i64 %53, 1
  br label %52

86:                                               ; preds = %52
  %87 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9) to i64))
  %88 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %87, 0
  %89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, ptr %87, 1
  %90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, i64 0, 2
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, i64 3, 3, 0
  %92 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %91, i64 3, 3, 1
  %93 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, i64 3, 4, 0
  %94 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %93, i64 1, 4, 1
  br label %95

95:                                               ; preds = %116, %86
  %96 = phi i64 [ %117, %116 ], [ 0, %86 ]
  %97 = icmp slt i64 %96, 3
  br i1 %97, label %98, label %118

98:                                               ; preds = %95
  br label %99

99:                                               ; preds = %102, %98
  %100 = phi i64 [ %115, %102 ], [ 0, %98 ]
  %101 = icmp slt i64 %100, 3
  br i1 %101, label %102, label %116

102:                                              ; preds = %99
  %103 = mul i64 %96, 3
  %104 = add i64 %103, %100
  %105 = getelementptr float, ptr %29, i64 %104
  %106 = load float, ptr %105, align 4
  %107 = mul i64 %96, 3
  %108 = add i64 %107, %100
  %109 = getelementptr float, ptr %8, i64 %108
  %110 = load float, ptr %109, align 4
  %111 = fadd float %106, %110
  %112 = mul i64 %96, 3
  %113 = add i64 %112, %100
  %114 = getelementptr float, ptr %87, i64 %113
  store float %111, ptr %114, align 4
  %115 = add i64 %100, 1
  br label %99

116:                                              ; preds = %99
  %117 = add i64 %96, 1
  br label %95

118:                                              ; preds = %95
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %94
}

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @simple_matmul(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %1, 1
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %7, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %8, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %9, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %10, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %12, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %11, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %13, 4, 1
  %29 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9) to i64))
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %29, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %29, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 0, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 3, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 3, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 3, 4, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 1, 4, 1
  br label %37

37:                                               ; preds = %49, %14
  %38 = phi i64 [ %50, %49 ], [ 0, %14 ]
  %39 = icmp slt i64 %38, 3
  br i1 %39, label %40, label %51

40:                                               ; preds = %37
  br label %41

41:                                               ; preds = %44, %40
  %42 = phi i64 [ %48, %44 ], [ 0, %40 ]
  %43 = icmp slt i64 %42, 3
  br i1 %43, label %44, label %49

44:                                               ; preds = %41
  %45 = mul i64 %38, 3
  %46 = add i64 %45, %42
  %47 = getelementptr float, ptr %29, i64 %46
  store float 0.000000e+00, ptr %47, align 4
  %48 = add i64 %42, 1
  br label %41

49:                                               ; preds = %41
  %50 = add i64 %38, 1
  br label %37

51:                                               ; preds = %37
  br label %52

52:                                               ; preds = %84, %51
  %53 = phi i64 [ %85, %84 ], [ 0, %51 ]
  %54 = icmp slt i64 %53, 3
  br i1 %54, label %55, label %86

55:                                               ; preds = %52
  br label %56

56:                                               ; preds = %82, %55
  %57 = phi i64 [ %83, %82 ], [ 0, %55 ]
  %58 = icmp slt i64 %57, 3
  br i1 %58, label %59, label %84

59:                                               ; preds = %56
  br label %60

60:                                               ; preds = %63, %59
  %61 = phi i64 [ %81, %63 ], [ 0, %59 ]
  %62 = icmp slt i64 %61, 3
  br i1 %62, label %63, label %82

63:                                               ; preds = %60
  %64 = mul i64 %53, 3
  %65 = add i64 %64, %61
  %66 = getelementptr float, ptr %1, i64 %65
  %67 = load float, ptr %66, align 4
  %68 = mul i64 %61, 3
  %69 = add i64 %68, %57
  %70 = getelementptr float, ptr %8, i64 %69
  %71 = load float, ptr %70, align 4
  %72 = mul i64 %53, 3
  %73 = add i64 %72, %57
  %74 = getelementptr float, ptr %29, i64 %73
  %75 = load float, ptr %74, align 4
  %76 = fmul float %67, %71
  %77 = fadd float %75, %76
  %78 = mul i64 %53, 3
  %79 = add i64 %78, %57
  %80 = getelementptr float, ptr %29, i64 %79
  store float %77, ptr %80, align 4
  %81 = add i64 %61, 1
  br label %60

82:                                               ; preds = %60
  %83 = add i64 %57, 1
  br label %56

84:                                               ; preds = %56
  %85 = add i64 %53, 1
  br label %52

86:                                               ; preds = %52
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %36
}

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @transpose_chain(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %1, 1
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 %2, 2
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 %3, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 %5, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 %4, 3, 1
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %6, 4, 1
  %15 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9) to i64))
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %15, 0
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 0, 2
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 3, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 3, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 3, 4, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 1, 4, 1
  br label %23

23:                                               ; preds = %39, %7
  %24 = phi i64 [ %40, %39 ], [ 0, %7 ]
  %25 = icmp slt i64 %24, 3
  br i1 %25, label %26, label %41

26:                                               ; preds = %23
  br label %27

27:                                               ; preds = %30, %26
  %28 = phi i64 [ %38, %30 ], [ 0, %26 ]
  %29 = icmp slt i64 %28, 3
  br i1 %29, label %30, label %39

30:                                               ; preds = %27
  %31 = mul i64 %24, 3
  %32 = add i64 %31, %28
  %33 = getelementptr float, ptr %1, i64 %32
  %34 = load float, ptr %33, align 4
  %35 = mul i64 %28, 3
  %36 = add i64 %35, %24
  %37 = getelementptr float, ptr %15, i64 %36
  store float %34, ptr %37, align 4
  %38 = add i64 %28, 1
  br label %27

39:                                               ; preds = %27
  %40 = add i64 %24, 1
  br label %23

41:                                               ; preds = %23
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %22
}

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @scalar_operations(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %1, 1
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 %2, 2
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 %3, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 %5, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 %4, 3, 1
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %6, 4, 1
  %15 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9) to i64))
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %15, 0
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 0, 2
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 3, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 3, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 3, 4, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 1, 4, 1
  br label %23

23:                                               ; preds = %40, %7
  %24 = phi i64 [ %41, %40 ], [ 0, %7 ]
  %25 = icmp slt i64 %24, 3
  br i1 %25, label %26, label %42

26:                                               ; preds = %23
  br label %27

27:                                               ; preds = %30, %26
  %28 = phi i64 [ %39, %30 ], [ 0, %26 ]
  %29 = icmp slt i64 %28, 3
  br i1 %29, label %30, label %40

30:                                               ; preds = %27
  %31 = mul i64 %24, 3
  %32 = add i64 %31, %28
  %33 = getelementptr float, ptr %1, i64 %32
  %34 = load float, ptr %33, align 4
  %35 = fmul float %34, 2.000000e+00
  %36 = mul i64 %24, 3
  %37 = add i64 %36, %28
  %38 = getelementptr float, ptr %15, i64 %37
  store float %35, ptr %38, align 4
  %39 = add i64 %28, 1
  br label %27

40:                                               ; preds = %27
  %41 = add i64 %24, 1
  br label %23

42:                                               ; preds = %23
  %43 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9) to i64))
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %43, 0
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, ptr %43, 1
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 0, 2
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 3, 3, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 3, 3, 1
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 3, 4, 0
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 1, 4, 1
  br label %51

51:                                               ; preds = %68, %42
  %52 = phi i64 [ %69, %68 ], [ 0, %42 ]
  %53 = icmp slt i64 %52, 3
  br i1 %53, label %54, label %70

54:                                               ; preds = %51
  br label %55

55:                                               ; preds = %58, %54
  %56 = phi i64 [ %67, %58 ], [ 0, %54 ]
  %57 = icmp slt i64 %56, 3
  br i1 %57, label %58, label %68

58:                                               ; preds = %55
  %59 = mul i64 %52, 3
  %60 = add i64 %59, %56
  %61 = getelementptr float, ptr %15, i64 %60
  %62 = load float, ptr %61, align 4
  %63 = fmul float %62, 5.000000e-01
  %64 = mul i64 %52, 3
  %65 = add i64 %64, %56
  %66 = getelementptr float, ptr %43, i64 %65
  store float %63, ptr %66, align 4
  %67 = add i64 %56, 1
  br label %55

68:                                               ; preds = %55
  %69 = add i64 %52, 1
  br label %51

70:                                               ; preds = %51
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %50
}

define void @main() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9) to i64))
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 3, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 3, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 3, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  %9 = getelementptr float, ptr %1, i64 0
  store float 1.000000e+00, ptr %9, align 4
  %10 = getelementptr float, ptr %1, i64 1
  store float 2.000000e+00, ptr %10, align 4
  %11 = getelementptr float, ptr %1, i64 2
  store float 3.000000e+00, ptr %11, align 4
  %12 = getelementptr float, ptr %1, i64 3
  store float 4.000000e+00, ptr %12, align 4
  %13 = getelementptr float, ptr %1, i64 4
  store float 5.000000e+00, ptr %13, align 4
  %14 = getelementptr float, ptr %1, i64 5
  store float 6.000000e+00, ptr %14, align 4
  %15 = getelementptr float, ptr %1, i64 6
  store float 7.000000e+00, ptr %15, align 4
  %16 = getelementptr float, ptr %1, i64 7
  store float 8.000000e+00, ptr %16, align 4
  %17 = getelementptr float, ptr %1, i64 8
  store float 9.000000e+00, ptr %17, align 4
  %18 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9) to i64))
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %18, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, ptr %18, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 0, 2
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 3, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 3, 3, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 3, 4, 0
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 1, 4, 1
  %26 = getelementptr float, ptr %18, i64 0
  store float 9.000000e+00, ptr %26, align 4
  %27 = getelementptr float, ptr %18, i64 1
  store float 8.000000e+00, ptr %27, align 4
  %28 = getelementptr float, ptr %18, i64 2
  store float 7.000000e+00, ptr %28, align 4
  %29 = getelementptr float, ptr %18, i64 3
  store float 6.000000e+00, ptr %29, align 4
  %30 = getelementptr float, ptr %18, i64 4
  store float 5.000000e+00, ptr %30, align 4
  %31 = getelementptr float, ptr %18, i64 5
  store float 4.000000e+00, ptr %31, align 4
  %32 = getelementptr float, ptr %18, i64 6
  store float 3.000000e+00, ptr %32, align 4
  %33 = getelementptr float, ptr %18, i64 7
  store float 2.000000e+00, ptr %33, align 4
  %34 = getelementptr float, ptr %18, i64 8
  store float 1.000000e+00, ptr %34, align 4
  %35 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0
  %36 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %37 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 2
  %38 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 3, 0
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 3, 1
  %40 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 4, 0
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 4, 1
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 0
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 1
  %44 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 2
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3, 0
  %46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 3, 1
  %47 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 4, 0
  %48 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, 4, 1
  %49 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @matrix_computation(ptr %35, ptr %36, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, ptr %42, ptr %43, i64 %44, i64 %45, i64 %46, i64 %47, i64 %48)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
