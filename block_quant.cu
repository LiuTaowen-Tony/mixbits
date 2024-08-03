#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <climits>
#include <tuple>
#include <cuda_bf16.h>
#include <torch/extension.h>



#define hd __host__ __device__
#define il __forceinline__
#define d_inline __device__ __forceinline__

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


__host__ d_inline int cdiv (int a, int b) { return (a + b - 1) / b; }

d_inline uint32_t reinterpret(float num) { return *(reinterpret_cast<uint32_t*>(&num)); }

d_inline float reinterpret_back(uint32_t num) { return *(reinterpret_cast<float*>(&num)); }

template <typename T>
d_inline float load_as_fp32(const T* ptr, int offset) { return *(ptr + offset); }

template <>
d_inline float load_as_fp32<__nv_bfloat16>(const __nv_bfloat16* ptr, int offset) { return __bfloat162float(*(ptr + offset)); }

template <typename T>
d_inline void store_from_fp32(T* ptr, float val, int offset) { *(ptr + offset) = val; }

template <>
d_inline void store_from_fp32<__nv_bfloat16>(__nv_bfloat16* ptr, float val, int offset) { *(ptr + offset) = __float2bfloat16(val); }

d_inline uint32_t man_round(
  uint32_t target,
  int man_bits,
  uint32_t rand_prob
) {
  uint32_t mask = (1 << (23-man_bits)) - 1;
  uint32_t add_r = target+(rand_prob & mask);
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

d_inline uint32_t clip_exp(
  uint32_t man_rounded,
  int exp_width,
  int man_width,
  uint32_t old_num
) {
  if (man_rounded == 0)
    return 0;
  int og_exp = man_rounded << 1 >> 1 >> 23;
  int max_exp = (1 << (exp_width - 1)) + 127;

  // compare using signed int
  if (og_exp <= max_exp) 
    return man_rounded;

  uint32_t sign = old_num & (1u << 31);
  uint32_t max_man = ((uint32_t) -1) << 9 >> 9 >> (23 - man_width) << (23 - man_width);
  uint32_t max_num = ((uint32_t) max_exp) << 23 | max_man;
  return sign | max_num;
}


d_inline float single_quant(
  float num, 
  int man_width, 
  int exp_width, 
  float scaling_factor, 
  float zero_point, 
  uint32_t round_helper
) {
    num = num - zero_point;
    num = num / scaling_factor;

    uint32_t unum = reinterpret(num);
    int target_exp = (unum << 1 >> 1 >> 23) -127; 
    auto min_exp = -((1 << (exp_width - 1)) - 2);
    bool is_subnormal = target_exp < min_exp;
    if (is_subnormal) {
        int shift_bits = ((127+min_exp)<<23) | (unum >> 31 <<31);        
        unum += shift_bits;
        uint32_t man_rounded = man_round(unum, man_width, round_helper);
        man_rounded -= shift_bits;
        num = reinterpret_back(man_rounded);
    } else {
        uint32_t man_rounded = man_round(unum, man_width, round_helper);
        uint32_t exp_rounded = clip_exp(man_rounded, exp_width, man_width, unum);
        num = reinterpret_back(exp_rounded);
    }
    num = num * scaling_factor;
    num = num + zero_point;
    return num;
}


unsigned int previous_power_of_2(unsigned int n) {
  if (n == 0) return 0; // Edge case: no previous power of 2 for 0
  unsigned int power = 1;
  while (power <= n) {
      power <<= 1; // Equivalent to power *= 2
  }
  return power >> 1; // Equivalent to dividing by 2
}

// for block has more than 32 elements
template<
  typename RoundingHelperGetter,
  typename ScalingFactorZeroPointGetter
>
__global__ void mid_block_kernel(
  const float* __restrict__ a,
  uint32_t* __restrict__ rand,
  float* __restrict__ b,
  int block_m,
  int block_n,
  int bf_block_n,
  int M,
  int N,
  int man_width,
  int exp_width,
  int s_m,
  int s_n
)
{
  // one wrap quantize a block; one thread quantize a value
  int block_offset = blockIdx.x * block_m * N + blockIdx.y * block_n;
  int i = threadIdx.x;
  int j = threadIdx.y;

  // load block A and compute mask
  bool mask = true;
  mask &= blockIdx.x * block_m + i < M;
  mask &= blockIdx.y * block_n + j < N;
  float val = 0;
  if (mask) 
    val = load_as_fp32(a, block_offset + i * N + j);

  // Perform warp-level reduction to find the maximum value in this block
  ScalingFactorZeroPointGetter sfzp;
  float zero_point = 0.0;
  float scaling_factor = 0.0;
  sfzp(i, j, block_m, block_n, bf_block_n, s_m, s_n, val, &scaling_factor, &zero_point);

  // quantize and store back
  if (! mask) 
    return;
  RoundingHelperGetter round_helper_getter;
  uint32_t round_helper = round_helper_getter(man_width, rand, block_offset + i * N + j);
  float out = single_quant(val, man_width, exp_width, scaling_factor, 0.0, round_helper);
  store_from_fp32(b, out, block_offset + i * N + j);
}

struct max_scaling_factor_helper {
  d_inline void operator()(
    int i, int j,
    int block_m, int block_n,
    int bf_block_n,
    int s_m, int s_n,
    float val,
    float* scaling_factor_out,
    float* zero_point_out
  ) {
    int jth_bf = j / bf_block_n;
    int bf_left_j = jth_bf * bf_block_n;
    int bf_right_j = (jth_bf + 1) * bf_block_n;
    __shared__ float sd[256];
    sd[i * block_m + j] = val;
    __syncthreads();
    for (int s = s_m; s > 0; s /= 2) {
      if (i < s && i + s < block_m)
        sd[i * block_m + j] = std::max(abs(sd[i * block_m + j]), abs(sd[(i + s) * block_m + j]));
      __syncthreads(); // always cross wrap
    }
    if (i == 0) {
      for (int s = s_n; s > 0; s /= 2) {
        if (j < s && j + s < bf_right_j) 
          sd[j] = std::max(abs(sd[j]), abs(sd[j + s]));
        __syncthreads();
      }
    }
    *scaling_factor_out = sd[bf_left_j];
    *zero_point_out = 0.0;
  }
};

struct const_scaling_factor_helper {
  d_inline void operator()(
    int i, int j,
    int block_m, int block_n,
    int bf_block_n,
    int s_m, int s_n,
    float val,
    float* scaling_factor_out,
    float* zero_point_out
  ) {
    *scaling_factor_out = 1.0;
    *zero_point_out = 0.0;
  }
};

struct const_rand_helper {
  d_inline uint32_t operator()(int man_width, const uint32_t* rand, int offset) { 
    return 1 << (23 - man_width - 1); 
  }
};

struct load_rand_helper {
  d_inline uint32_t operator()(int man_width, const uint32_t* rand, int offset) { return rand[offset]; }
};





torch::Tensor block_quant(
  torch::Tensor a,
  int man_width, int exp_width,
  int bf_block_m, int bf_block_n,
  bool is_stochastic_round,
  bool is_max_scaling
)
{
  CHECK_INPUT(a);
  cudaSetDevice(a.get_device());
  auto b = torch::zeros_like(a);
  uint32_t* rand_ptr = nullptr;
  if (is_stochastic_round) {
    torch::Tensor rand = torch::randint_like(a, INT_MAX, 
      torch::device(torch::kCUDA).dtype(torch::kUInt32));
    rand_ptr = rand.data_ptr<uint32_t>();
  }
  int M = a.size(0);
  int N = a.size(1);
  int block_m = bf_block_m;
  int block_n = std::lcm(bf_block_n, 4);
  int s_m = previous_power_of_2(block_m);
  int s_n = previous_power_of_2(block_n);

  if (is_stochastic_round && ! is_max_scaling) {
    mid_block_kernel<const_rand_helper, const_scaling_factor_helper><<<dim3(cdiv(M, block_m), cdiv(N, block_n)), dim3(block_m, block_n)>>>(
      a.data_ptr<float>(), rand_ptr, b.data_ptr<float>(), block_m, block_n, bf_block_n, M, N, man_width, exp_width, s_m, s_n
    );
  } else if (is_stochastic_round && is_max_scaling) {
    mid_block_kernel<load_rand_helper, max_scaling_factor_helper><<<dim3(cdiv(M, block_m), cdiv(N, block_n)), dim3(block_m, block_n)>>>(
      a.data_ptr<float>(), rand_ptr, b.data_ptr<float>(), block_m, block_n, bf_block_n, M, N, man_width, exp_width, s_m, s_n
    );
  } else if (! is_stochastic_round && ! is_max_scaling) {
    mid_block_kernel<const_rand_helper, const_scaling_factor_helper><<<dim3(cdiv(M, block_m), cdiv(N, block_n)), dim3(block_m, block_n)>>>(
      a.data_ptr<float>(), nullptr, b.data_ptr<float>(), block_m, block_n, bf_block_n, M, N, man_width, exp_width, s_m, s_n
    );
  } else {
    mid_block_kernel<load_rand_helper, max_scaling_factor_helper><<<dim3(cdiv(M, block_m), cdiv(N, block_n)), dim3(block_m, block_n)>>>(
      a.data_ptr<float>(), nullptr, b.data_ptr<float>(), block_m, block_n, bf_block_n, M, N, man_width, exp_width, s_m, s_n
    );
  }

  return b;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_quant", &block_quant, "block quantization");
}
