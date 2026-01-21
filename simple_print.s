	.text
	.file	"LLVMDialectModule"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
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
	movl	$16, %edi
	callq	malloc@PLT
	movq	%rax, %rbx
	movabsq	$4611686019492741120, %rax      # imm = 0x400000003F800000
	movq	%rax, (%rbx)
	movabsq	$4647714816524288000, %rax      # imm = 0x4080000040400000
	movq	%rax, 8(%rbx)
	xorl	%r14d, %r14d
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_5:                                #   in Loop: Header=BB0_1 Depth=1
	callq	print_newline@PLT
	incq	%r14
	addq	$8, %rbx
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	cmpq	$1, %r14
	jg	.LBB0_6
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	xorl	%r15d, %r15d
	cmpq	$1, %r15
	jg	.LBB0_5
	.p2align	4, 0x90
.LBB0_4:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%rbx,%r15,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	callq	print_f32@PLT
	incq	%r15
	cmpq	$1, %r15
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
