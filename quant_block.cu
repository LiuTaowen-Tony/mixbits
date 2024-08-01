#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>
#include <tuple>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define hd __host__ __device__
#define il __forceinline__
#define hd_inline __host__ __device__ __forceinline__

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

hd_inline int cdiv (int a, int b) {
    return (a + b - 1) / b;
}


hd_inline uint32_t reinterpret(float num) {
    return *(reinterpret_cast<uint32_t*>(&num));
}

hd_inline float reinterpret_back(uint32_t num) {
    return *(reinterpret_cast<float*>(&num));
}

// hd_inline uint32_t get_sign(float num) {
//     uint32_t unum = reinterpret(num);
//     return unum >> 31;
// }

// hd_inline uint32_t get_exp(float num) {
//     uint32_t unum = reinterpret(num);
//     return unum << 1 >> 24;
// }

// hd_inline uint32_t get_man(float num) {
//     uint32_t unum = reinterpret(num);
//     return unum << 9 >> 9;
// }

hd_inline uint32_t man_round(
  uint32_t target,
  int man_bits,
  uint32_t rand_prob
) {
  uint32_t mask = (1 << (23-man_bits)) - 1;
  uint32_t add_r = target+(rand_prob & mask);
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

hd_inline uint32_t clip_exp(
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


hd_inline float single_quant(float num, int man_width, int exp_width, float scaling_factor, float zero_point, uint32_t round_helper) {
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

template <typename T, int NUM>
__inline__ __device__ T warpReduceMax(T* val, int thread_group_width = 32) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
      val[i] = max(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, 32));
    }
  }
  return (T)(0.0f);
}

template <typename T>
hd_inline float type_convert_load(const void* ptr, int offset);

template <>
hd_inline float type_convert_load<float>(const void* ptr, size_t offset) {
  return *(((float*) ptr) + offset);
}

// template <>
// hd_inline float type_convert_load<__bfloat>(const void* ptr, size_t offset) {
//   return __bfloat162float(*(((__nv_bfloat16*) ptr) + offset));
// }

template <typename T>
hd_inline void type_convert_store(void* ptr, float val);

template <>
hd_inline void type_convert_store<float>(void* ptr, float val, size_t offset) {
  *(((float*) ptr) + offset) = val;
}

// template <>
// hd_inline void type_convert_store<__bfloat>(void* ptr, float val, size_t offset) {
//   *(((__nv_bfloat16*) ptr) + offset) = __float2bfloat16(val);
// }

// for block has more than 32 elements
template <typename TensorDType>
__global__ void big_block_kernel(
  const void* __restrict__ a,
  uint32_t* __restrict__ rand,
  void* __restrict__ b,
  int block_m,
  int block_n,
  int M,
  int N,
  int man_width,
  int exp_width,
  bool is_stochastic_round
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
    val = type_convert_load<TensorDType>(a, block_offset + i * N + j);

  // Perform warp-level reduction to find the maximum value in this block
  float scaling_factor = val;
  for (int offset = block_m * block_n / 2; offset > 0; offset /= 2) {
      scaling_factor = max(scaling_factor, __shfl_down_sync(0xFFFFFFFF, scaling_factor, offset));
  }
  __syncthreads();
  
  // quantize and store back
  if (! mask) { return; }
  uint32_t round_helper = (1 << (23 - man_width)) - 1;
  if (is_stochastic_round) { round_helper = rand[block_offset + i * N + j]; }
  float out = single_quant(val, man_width, exp_width, scaling_factor, 0.0, round_helper);
  type_convert_store<TensorDType>(b, out, block_offset + i * N + j);
}

template <typename TensorDType>
__global__ void small_block_kernel(
  const void* __restrict__ a,
  uint32_t* __restrict__ rand,
  void* __restrict__ b,
  int block_m,
  int block_n,
  int M,
  int N,
  int man_width,
  int exp_width,
  bool is_stochastic_round,
)
{
  // when block is small one thread quantize a block
  int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  int pidy = blockIdx.y * blockDim.y + threadIdx.y;
  int block_offset = pidx * block_m * N + pidy * block_n;  

  // load block A, smaller than 32 elements
  float block_a[32];
  for (int i = 0; i < block_m; i++) {
    for (int j = 0; j < block_n; j++) {
      if (pidx * block_m + i < M && pidy * block_n + j < N) { 
        block_a[i * block_n + j] = type_convert_load<TensorDType>(a, block_offset + i * N + j);
      } else {
        block_a[i * block_n + j] = 0;
      }
    }
  }

  // find the maximum value in this block
  float scaling_factor = -1e9;
  for (int i = 0; i < block_m; i++) {
    for (int j = 0; j < block_n; j++) {
      scaling_factor = max(scaling_factor, block_a[i * block_n + j]);
    }
  }
  
  uint32_t round_helper = (1 << (23 - man_width)) - 1;

  // quantize and store back
  for (int i = 0; i < block_m; i++) {
    for (int j = 0; j < block_n; j++) {
      if (pidx * block_m + i < M && pidy * block_n + j < N) {
        float out = single_quant(block_a[i * block_n + j], man_width, exp_width, scaling_factor, 0.0, round_helper);
        type_convert_store<TensorDType>(b, out, block_offset + i * N + j);
      }
    }
  } 
}

at::Tensor block_quant(
  at::Tensor a,
  int man_width, int exp_width,
  int block_m, int block_n,
  bool is_stochastic_round
)
{
  cudaSetDevice(a.get_device());
  auto b = at::zeros_like(a);
  at::Tensor rand;
  if (is_stochastic_round)
    rand = at::rand_like(a);
  int block_size = block_m * block_n;
  int M = a.size(0);
  int N = a.size(1);
  auto type_convert = 0;

  if (block_size < 32) {
    dim3 block(4, 8);
    dim3 grid(cdiv(M, block_m * 4), cdiv(N, block_n * 8));
    using TensorDType = float;
    small_block_kernel<TensorDType><<<grid, block>>>(
      a.data_ptr<void>(), rand.data_ptr<uint32_t>(), b.data_ptr<void>(),
      block_m, block_n,
      M, N,
      man_width, exp_width,
      is_stochastic_round,
    );
  } else {
    dim3 block(block_m, block_n);
    dim3 grid(cdiv(M, block_m), cdiv(N, block_n));
    big_block_kernel<<<grid, block>>>(
      a.data_ptr<float>(), rand.data_ptr<uint32_t>(), b.data_ptr<float>(),
      block_m, block_n,
      M, N,
      man_width, exp_width,
      is_stochastic_round,
    );
  }
  return b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_quant", &block_quant, "block quantization");
}