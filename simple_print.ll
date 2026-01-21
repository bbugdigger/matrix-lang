; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @print_f32(float)

declare void @print_newline()

define void @main() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4) to i64))
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 2, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 2, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 2, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  %9 = getelementptr float, ptr %1, i64 0
  store float 1.000000e+00, ptr %9, align 4
  %10 = getelementptr float, ptr %1, i64 1
  store float 2.000000e+00, ptr %10, align 4
  %11 = getelementptr float, ptr %1, i64 2
  store float 3.000000e+00, ptr %11, align 4
  %12 = getelementptr float, ptr %1, i64 3
  store float 4.000000e+00, ptr %12, align 4
  br label %13

13:                                               ; preds = %26, %0
  %14 = phi i64 [ %27, %26 ], [ 0, %0 ]
  %15 = icmp slt i64 %14, 2
  br i1 %15, label %16, label %28

16:                                               ; preds = %13
  br label %17

17:                                               ; preds = %20, %16
  %18 = phi i64 [ %25, %20 ], [ 0, %16 ]
  %19 = icmp slt i64 %18, 2
  br i1 %19, label %20, label %26

20:                                               ; preds = %17
  %21 = mul i64 %14, 2
  %22 = add i64 %21, %18
  %23 = getelementptr float, ptr %1, i64 %22
  %24 = load float, ptr %23, align 4
  call void @print_f32(float %24)
  %25 = add i64 %18, 1
  br label %17

26:                                               ; preds = %17
  call void @print_newline()
  %27 = add i64 %14, 1
  br label %13

28:                                               ; preds = %13
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
