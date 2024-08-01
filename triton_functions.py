import torch
import numpy as np
import triton
import triton.language as tl

MAX_WIDTH = 16
MAX_MAN_WIDTH = 7
MAX_EXP_WIDTH = 8
SIGN_WIDTH = 1

@triton.jit
def get_bf_exp_as_int8(num: tl.bfloat16) -> tl.int8:
    # unum = tl.uint16(num)
    unum = num.to(tl.uint16, bitcast=True)
    unum = unum << SIGN_WIDTH >> (MAX_EXP_WIDTH + 1)
    return num.to(tl.int8)

@triton.jit
def get_bf_sign_as_bool(num: tl.bfloat16) -> tl.uint16:
    unum = num.to(tl.uint16, bitcast=True)
    return unum >> (MAX_WIDTH - SIGN_WIDTH)

@triton.jit
def get_bf_man(num: tl.bfloat16) -> tl.uint16:
    unum = num.to(tl.uint16, bitcast=True)
    return unum << (SIGN_WIDTH + MAX_EXP_WIDTH) >> (SIGN_WIDTH + MAX_EXP_WIDTH)

@triton.jit
def man_round(unum: tl.uint16, man_width: int, round_helper: tl.uint16) -> tl.uint16:
    mask = tl.uint16(1 << (7 - man_width)) - 1
    rand_prob = (round_helper & mask)
    add_result = unum + rand_prob
    quantized = add_result & ~mask
    return quantized

@triton.jit
def clip_exp(man_rounded: tl.uint16, 
              exp_width: int,
              man_width: int, 
              old_num: tl.uint16) -> tl.uint16:
    if man_rounded == 0:
        return man_rounded
    og_exp = get_bf_exp_as_int8(man_rounded)
    max_exp = (1 << (exp_width - 1)) - 1
    sign = get_bf_sign_as_bool(old_num) << 15
    if og_exp > max_exp:
        max_man = tl.uint16(0x7F) >> (7 - man_width) << (7 - man_width)
        max_num = tl.uint16(max_exp) << 7 | max_man
        quantized = sign | max_num
    return quantized

@triton.jit
def single_quant(num: tl.bfloat16, 
                 man_width: int, 
                 exp_width: int, 
                 scaling_factor: tl.bfloat16, 
                 zero_point: tl.bfloat16,
                 round_helper: tl.uint16,):
    num = (num - zero_point)
    num = num / scaling_factor

    target_exp = get_bf_exp_as_int8(num)
    unum = num.to(tl.uint16, bitcast=True)
    min_exp = -((1 << (exp_width - 1)) - 2)
    is_subnormal = target_exp < min_exp
    if is_subnormal:
        shift_bits = (127 + min_exp) << 7 | (unum & 0x7000)
        unum += shift_bits
        man_rounded = man_round(unum, man_width, round_helper)
        man_rounded -= shift_bits
        num = man_rounded.to(tl.bfloat16, bitcast=True)
    else:
        man_rounded = man_round(unum, man_width, round_helper)
        exp_rounded = clip_exp(man_rounded, exp_width, man_width, unum)
        num = exp_rounded.to(tl.bfloat16, bitcast=True)
    num = num * scaling_factor
    num = num + zero_point
    return num

@triton.jit
def block_quant(
    A, B, R,
    mat_block_m: tl.constexpr, mat_block_n: tl.constexpr,
    bf_block_m: tl.constexpr, bf_block_n: tl.constexpr,
    grid_block_m: tl.constexpr, grid_block_n: tl.constexpr,
    M, N,
    man_width, exp_width,
    stochastic_rounding_flag,
):
    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    mat_block_offset = pidx * mat_block_m * M + pidy * mat_block_n

    offsets_m = tl.arange(0, bf_block_m)[:, None]
    offsets_n = tl.arange(0, bf_block_n)


    quant_helper_block = tl.zeros((bf_block_m, bf_block_n), dtype=tl.uint16)
    quant_helper_block += (1 << (MAX_MAN_WIDTH - man_width - 1)).to(tl.uint16)
    for i in range(grid_block_m):
        for j in range(grid_block_n):
            bf_block_offset = i * bf_block_m * M + j * bf_block_n
            A_block = A + mat_block_offset + bf_block_offset + offsets_m * N + offsets_n
            B_block = B + mat_block_offset + bf_block_offset + offsets_m * N + offsets_n
            R_block = R + mat_block_offset + bf_block_offset + offsets_m * N + offsets_n

            block_start_m = pidx * mat_block_m + i * bf_block_m
            block_start_n = pidy * mat_block_n + j * bf_block_n
            mask_m = (block_start_m + offsets_m) < M
            mask_n = (block_start_n + offsets_n) < N
            block_mask_m = offsets_m < (mat_block_m - i * bf_block_m)
            block_mask_n = offsets_n < (mat_block_n - j * bf_block_n)
            mask = mask_m & mask_n & block_mask_m & block_mask_n

            if stochastic_rounding_flag:
                quant_helper_block = tl.load(R_block, mask=mask)

            A_vals = tl.load(A_block, mask=mask)
            scale = tl.max(tl.abs(A_vals)).to(tl.bfloat16)

            # Apply quantization within a mapped function or loop
            for k in range(bf_block_m):
                for l in range(bf_block_n):
                    if mask[k, l]:
                        A_vals[k, l] = single_quant(A_vals[k, l], man_width, exp_width, scale, 0.0, quant_helper_block[k, l])

            tl.store(B_block, A_vals, mask=mask)


@triton.jit
def clamp(a, min, max):
    return a if a > min else min if a < min else max if a > max else a

#@triton.jit
#def 


def launch_block_quant(A, B, mat_block_m, mat_block_n, bf_block_m, bf_block_n, man_width, exp_width, stochastic_rounding_flag):
    M, N = A.shape
    grid = (triton.cdiv(M, mat_block_m), triton.cdiv(N, mat_block_n))
    grid_block_m = triton.cdiv(mat_block_m, bf_block_m)
    grid_block_n = triton.cdiv(mat_block_n, bf_block_n)
    R = torch.randint_like(A, high=2**16-1 ,dtype=torch.uint16, device=A.device)
    # Kernel launch
    block_quant[grid](
        A, B, R,
        mat_block_m, mat_block_n,
        bf_block_m, bf_block_n,
        grid_block_m, grid_block_n,
        M, N,
        man_width, exp_width,
        stochastic_rounding_flag
    )

# Example matrices
M = 1024
N = 1024
mat_block_m = 128
mat_block_n = 128
bf_block_m = 16
bf_block_n = 16
man_width = 7
exp_width = 8
stochastic_rounding_flag = True

A = torch.randn(M, N, device='cuda')
B = torch.empty_like(A)

# Launch the kernel
launch_block_quant(A, B, mat_block_m, mat_block_n, bf_block_m, bf_block_n, man_width, exp_width, stochastic_rounding_flag)

print(A)
print(B)
