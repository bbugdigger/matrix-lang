	.text
	.file	"LLVMDialectModule"
	.globl	test_add                        # -- Begin function test_add
	.p2align	4, 0x90
	.type	test_add,@function
test_add:                               # @test_add
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
	movq	%rdx, %r14
	movq	%rdi, %rbx
	movq	56(%rsp), %r15
	movl	$36, %edi
	callq	malloc@PLT
	movl	$1, %ecx
	movl	$3, %edx
	xorl	%esi, %esi
	movq	%rax, %rdi
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_5:                                #   in Loop: Header=BB0_1 Depth=1
	incq	%rsi
	addq	$12, %rdi
	addq	$12, %r15
	addq	$12, %r14
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	cmpq	$2, %rsi
	jg	.LBB0_6
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	xorl	%r8d, %r8d
	cmpq	$2, %r8
	jg	.LBB0_5
	.p2align	4, 0x90
.LBB0_4:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%r14,%r8,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	addss	(%r15,%r8,4), %xmm0
	movss	%xmm0, (%rdi,%r8,4)
	incq	%r8
	cmpq	$2, %r8
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
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
.Lfunc_end0:
	.size	test_add, .Lfunc_end0-test_add
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
