	.text
	.file	"LLVMDialectModule"
	.globl	matrix_computation              # -- Begin function matrix_computation
	.p2align	4, 0x90
	.type	matrix_computation,@function
matrix_computation:                     # @matrix_computation
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %r15
	movq	%rdi, %rbx
	movq	72(%rsp), %r12
	movl	$36, %edi
	callq	malloc@PLT
	movq	%rax, %r14
	xorl	%eax, %eax
	movq	%r14, %rcx
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_5:                                #   in Loop: Header=BB0_1 Depth=1
	incq	%rax
	addq	$12, %rcx
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	cmpq	$2, %rax
	jg	.LBB0_6
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	xorl	%edx, %edx
	cmpq	$2, %rdx
	jg	.LBB0_5
	.p2align	4, 0x90
.LBB0_4:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	$0, (%rcx,%rdx,4)
	incq	%rdx
	cmpq	$2, %rdx
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
	xorl	%eax, %eax
	jmp	.LBB0_7
	.p2align	4, 0x90
.LBB0_14:                               #   in Loop: Header=BB0_7 Depth=1
	incq	%rax
	addq	$12, %r15
.LBB0_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_9 Depth 2
                                        #       Child Loop BB0_12 Depth 3
	cmpq	$2, %rax
	jg	.LBB0_15
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=1
	leaq	(%rax,%rax,2), %rcx
	movq	%r12, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_9
	.p2align	4, 0x90
.LBB0_13:                               #   in Loop: Header=BB0_9 Depth=2
	incq	%rsi
	addq	$4, %rdx
.LBB0_9:                                #   Parent Loop BB0_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_12 Depth 3
	cmpq	$2, %rsi
	jg	.LBB0_14
# %bb.10:                               #   in Loop: Header=BB0_9 Depth=2
	leaq	(%rcx,%rsi), %rdi
	movq	%rdx, %r8
	xorl	%r9d, %r9d
	cmpq	$2, %r9
	jg	.LBB0_13
	.p2align	4, 0x90
.LBB0_12:                               #   Parent Loop BB0_7 Depth=1
                                        #     Parent Loop BB0_9 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	(%r15,%r9,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	mulss	(%r8), %xmm0
	addss	(%r14,%rdi,4), %xmm0
	movss	%xmm0, (%r14,%rdi,4)
	incq	%r9
	addq	$12, %r8
	cmpq	$2, %r9
	jle	.LBB0_12
	jmp	.LBB0_13
.LBB0_15:
	movl	$36, %edi
	callq	malloc@PLT
	movl	$1, %ecx
	movl	$3, %edx
	xorl	%esi, %esi
	movq	%rax, %rdi
	jmp	.LBB0_16
	.p2align	4, 0x90
.LBB0_20:                               #   in Loop: Header=BB0_16 Depth=1
	incq	%rsi
	addq	$12, %rdi
	addq	$12, %r12
	addq	$12, %r14
.LBB0_16:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_19 Depth 2
	cmpq	$2, %rsi
	jg	.LBB0_21
# %bb.17:                               #   in Loop: Header=BB0_16 Depth=1
	xorl	%r8d, %r8d
	cmpq	$2, %r8
	jg	.LBB0_20
	.p2align	4, 0x90
.LBB0_19:                               #   Parent Loop BB0_16 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%r14,%r8,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	addss	(%r12,%r8,4), %xmm0
	movss	%xmm0, (%rdi,%r8,4)
	incq	%r8
	cmpq	$2, %r8
	jle	.LBB0_19
	jmp	.LBB0_20
.LBB0_21:
	movq	%rax, (%rbx)
	movq	%rax, 8(%rbx)
	movq	%rdx, 24(%rbx)
	movq	%rdx, 32(%rbx)
	movq	%rdx, 40(%rbx)
	movq	%rcx, 48(%rbx)
	movq	$0, 16(%rbx)
	movq	%rbx, %rax
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	matrix_computation, .Lfunc_end0-matrix_computation
	.cfi_endproc
                                        # -- End function
	.globl	simple_matmul                   # -- Begin function simple_matmul
	.p2align	4, 0x90
	.type	simple_matmul,@function
simple_matmul:                          # @simple_matmul
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %r14
	movq	%rdi, %rbx
	movq	72(%rsp), %r15
	movl	$36, %edi
	callq	malloc@PLT
	movl	$1, %ecx
	movl	$3, %edx
	xorl	%esi, %esi
	movq	%rax, %rdi
	jmp	.LBB1_1
	.p2align	4, 0x90
.LBB1_5:                                #   in Loop: Header=BB1_1 Depth=1
	incq	%rsi
	addq	$12, %rdi
.LBB1_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_4 Depth 2
	cmpq	$2, %rsi
	jg	.LBB1_6
# %bb.2:                                #   in Loop: Header=BB1_1 Depth=1
	xorl	%r8d, %r8d
	cmpq	$2, %r8
	jg	.LBB1_5
	.p2align	4, 0x90
.LBB1_4:                                #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	$0, (%rdi,%r8,4)
	incq	%r8
	cmpq	$2, %r8
	jle	.LBB1_4
	jmp	.LBB1_5
.LBB1_6:
	xorl	%esi, %esi
	jmp	.LBB1_7
	.p2align	4, 0x90
.LBB1_14:                               #   in Loop: Header=BB1_7 Depth=1
	incq	%rsi
	addq	$12, %r14
.LBB1_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_9 Depth 2
                                        #       Child Loop BB1_12 Depth 3
	cmpq	$2, %rsi
	jg	.LBB1_15
# %bb.8:                                #   in Loop: Header=BB1_7 Depth=1
	leaq	(%rsi,%rsi,2), %rdi
	movq	%r15, %r8
	xorl	%r9d, %r9d
	jmp	.LBB1_9
	.p2align	4, 0x90
.LBB1_13:                               #   in Loop: Header=BB1_9 Depth=2
	incq	%r9
	addq	$4, %r8
.LBB1_9:                                #   Parent Loop BB1_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_12 Depth 3
	cmpq	$2, %r9
	jg	.LBB1_14
# %bb.10:                               #   in Loop: Header=BB1_9 Depth=2
	leaq	(%rdi,%r9), %r10
	movq	%r8, %r11
	xorl	%r12d, %r12d
	cmpq	$2, %r12
	jg	.LBB1_13
	.p2align	4, 0x90
.LBB1_12:                               #   Parent Loop BB1_7 Depth=1
                                        #     Parent Loop BB1_9 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	(%r14,%r12,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	mulss	(%r11), %xmm0
	addss	(%rax,%r10,4), %xmm0
	movss	%xmm0, (%rax,%r10,4)
	incq	%r12
	addq	$12, %r11
	cmpq	$2, %r12
	jle	.LBB1_12
	jmp	.LBB1_13
.LBB1_15:
	movq	%rax, (%rbx)
	movq	%rax, 8(%rbx)
	movq	%rdx, 24(%rbx)
	movq	%rdx, 32(%rbx)
	movq	%rdx, 40(%rbx)
	movq	%rcx, 48(%rbx)
	movq	$0, 16(%rbx)
	movq	%rbx, %rax
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	simple_matmul, .Lfunc_end1-simple_matmul
	.cfi_endproc
                                        # -- End function
	.globl	transpose_chain                 # -- Begin function transpose_chain
	.p2align	4, 0x90
	.type	transpose_chain,@function
transpose_chain:                        # @transpose_chain
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rdx, %r14
	movq	%rdi, %rbx
	movl	$36, %edi
	callq	malloc@PLT
	movl	$1, %ecx
	movl	$3, %edx
	xorl	%esi, %esi
	movq	%rax, %rdi
	jmp	.LBB2_1
	.p2align	4, 0x90
.LBB2_5:                                #   in Loop: Header=BB2_1 Depth=1
	incq	%rsi
	addq	$4, %rdi
	addq	$12, %r14
.LBB2_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_4 Depth 2
	cmpq	$2, %rsi
	jg	.LBB2_6
# %bb.2:                                #   in Loop: Header=BB2_1 Depth=1
	movq	%rdi, %r8
	xorl	%r9d, %r9d
	cmpq	$2, %r9
	jg	.LBB2_5
	.p2align	4, 0x90
.LBB2_4:                                #   Parent Loop BB2_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%r14,%r9,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%r8)
	incq	%r9
	addq	$12, %r8
	cmpq	$2, %r9
	jle	.LBB2_4
	jmp	.LBB2_5
.LBB2_6:
	movq	%rax, (%rbx)
	movq	%rax, 8(%rbx)
	movq	%rdx, 24(%rbx)
	movq	%rdx, 32(%rbx)
	movq	%rdx, 40(%rbx)
	movq	%rcx, 48(%rbx)
	movq	$0, 16(%rbx)
	movq	%rbx, %rax
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	transpose_chain, .Lfunc_end2-transpose_chain
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function scalar_operations
.LCPI3_0:
	.long	0x3f000000                      # float 0.5
	.text
	.globl	scalar_operations
	.p2align	4, 0x90
	.type	scalar_operations,@function
scalar_operations:                      # @scalar_operations
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %r15
	movq	%rdi, %rbx
	movl	$36, %edi
	callq	malloc@PLT
	movq	%rax, %r14
	xorl	%eax, %eax
	movq	%r14, %rcx
	jmp	.LBB3_1
	.p2align	4, 0x90
.LBB3_5:                                #   in Loop: Header=BB3_1 Depth=1
	incq	%rax
	addq	$12, %rcx
	addq	$12, %r15
.LBB3_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_4 Depth 2
	cmpq	$2, %rax
	jg	.LBB3_6
# %bb.2:                                #   in Loop: Header=BB3_1 Depth=1
	xorl	%edx, %edx
	cmpq	$2, %rdx
	jg	.LBB3_5
	.p2align	4, 0x90
.LBB3_4:                                #   Parent Loop BB3_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%r15,%rdx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	%xmm0, %xmm0
	movss	%xmm0, (%rcx,%rdx,4)
	incq	%rdx
	cmpq	$2, %rdx
	jle	.LBB3_4
	jmp	.LBB3_5
.LBB3_6:
	movl	$36, %edi
	callq	malloc@PLT
	movl	$1, %ecx
	movl	$3, %edx
	xorl	%esi, %esi
	movss	.LCPI3_0(%rip), %xmm0           # xmm0 = [5.0E-1,0.0E+0,0.0E+0,0.0E+0]
	movq	%rax, %rdi
	jmp	.LBB3_7
	.p2align	4, 0x90
.LBB3_11:                               #   in Loop: Header=BB3_7 Depth=1
	incq	%rsi
	addq	$12, %rdi
	addq	$12, %r14
.LBB3_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_10 Depth 2
	cmpq	$2, %rsi
	jg	.LBB3_12
# %bb.8:                                #   in Loop: Header=BB3_7 Depth=1
	xorl	%r8d, %r8d
	cmpq	$2, %r8
	jg	.LBB3_11
	.p2align	4, 0x90
.LBB3_10:                               #   Parent Loop BB3_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%r14,%r8,4), %xmm1             # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm0, %xmm1
	movss	%xmm1, (%rdi,%r8,4)
	incq	%r8
	cmpq	$2, %r8
	jle	.LBB3_10
	jmp	.LBB3_11
.LBB3_12:
	movq	%rax, (%rbx)
	movq	%rax, 8(%rbx)
	movq	%rdx, 24(%rbx)
	movq	%rdx, 32(%rbx)
	movq	%rdx, 40(%rbx)
	movq	%rcx, 48(%rbx)
	movq	$0, 16(%rbx)
	movq	%rbx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end3:
	.size	scalar_operations, .Lfunc_end3-scalar_operations
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$64, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -16
	movl	$36, %edi
	callq	malloc@PLT
	movq	%rax, %rbx
	movabsq	$4611686019492741120, %rax      # imm = 0x400000003F800000
	movq	%rax, (%rbx)
	movabsq	$4647714816524288000, %rax      # imm = 0x4080000040400000
	movq	%rax, 8(%rbx)
	movabsq	$4665729215040061440, %rax      # imm = 0x40C0000040A00000
	movq	%rax, 16(%rbx)
	movabsq	$4683743613553737728, %rax      # imm = 0x4100000040E00000
	movq	%rax, 24(%rbx)
	movl	$1091567616, 32(%rbx)           # imm = 0x41100000
	movl	$36, %edi
	callq	malloc@PLT
	movabsq	$4683743613556883456, %rcx      # imm = 0x4100000041100000
	movq	%rcx, (%rax)
	movabsq	$4665729215044255744, %rcx      # imm = 0x40C0000040E00000
	movq	%rcx, 8(%rax)
	movabsq	$4647714816530579456, %rcx      # imm = 0x4080000040A00000
	movq	%rcx, 16(%rax)
	movabsq	$4611686019505324032, %rcx      # imm = 0x4000000040400000
	movq	%rcx, 24(%rax)
	movl	$1065353216, 32(%rax)           # imm = 0x3F800000
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	16(%rsp), %rdi
	movl	$3, %r8d
	movl	$3, %r9d
	movq	%rbx, %rsi
	movq	%rbx, %rdx
	xorl	%ecx, %ecx
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$3
	.cfi_adjust_cfa_offset 8
	callq	matrix_computation@PLT
	addq	$144, %rsp
	.cfi_adjust_cfa_offset -144
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	main, .Lfunc_end4-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
