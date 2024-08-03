#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <climits>
#include <tuple>
#include <torch/torch.h>
#include <ATen/ATen.h>

#define hd __host__ __device__
#define il __forceinline__
#define d_inline __device__ __forceinline__

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


__host__ d_inline int cdiv (int a, int b) {
    return (a + b - 1) / b;
}


d_inline uint32_t reinterpret(float num) {
    return *(reinterpret_cast<uint32_t*>(&num));
}

d_inline float reinterpret_back(uint32_t num) {
    return *(reinterpret_cast<float*>(&num));
}

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


d_inline float single_quant(float num, int man_width, int exp_width, float scaling_factor, float zero_point, uint32_t round_helper) {
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

template <typename T>
d_inline float load_as_fp32(const T* ptr, int offset) {
  return *(ptr + offset);
}

template <>
d_inline float load_as_fp32<__nv_bfloat16>(const __nv_bfloat16* ptr, int offset) {
  return __bfloat162float(*(ptr + offset));
}

template <typename T>
d_inline void store_from_fp32(T* ptr, float val, int offset) {
  *(ptr + offset) = val;
}

template <>
d_inline void store_from_fp32<__nv_bfloat16>(__nv_bfloat16* ptr, float val, int offset) {
  *(ptr + offset) = __float2bfloat16(val);
}


// for block has more than 32 elements
template<
  typename LoadType, 
  typename StoreType,
>
__global__ void big_block_kernel(
  const LoadType* __restrict__ a,
  uint32_t* __restrict__ rand,
  StoreType* __restrict__ b,
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
  if (mask) { val = load_as_fp32(a, block_offset + i * N + j); }

  // Perform warp-level reduction to find the maximum value in this block
  float scaling_factor = val;
  for (int offset = block_m * block_n / 2; offset > 0; offset /= 2) {
      scaling_factor = max(scaling_factor, __shfl_down_sync(0xFFFFFFFF, scaling_factor, offset));
  }
  __syncthreads();
  __shared__ float shared_scaling_factor;
  if (i == 0 && j == 0) { shared_scaling_factor = scaling_factor; }
  __syncthreads();
  scaling_factor = shared_scaling_factor;
  
  // quantize and store back
  if (! mask) { return; }
  uint32_t round_helper = 1 << (23 - man_width - 1);
  if (is_stochastic_round) { round_helper = rand[block_offset + i * N + j]; }
  float out = single_quant(val, man_width, exp_width, scaling_factor, 0.0, round_helper);
  store_from_fp32(b, out, block_offset + i * N + j);
}

template<typename LoadType, typename StoreType>
__global__ void small_block_kernel(
  const LoadType* __restrict__ a,
  uint32_t* __restrict__ rand,
  StoreType* __restrict__ b,
  int block_m,
  int block_n,
  int M,
  int N,
  int man_width,
  int exp_width,
  bool is_stochastic_round
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
        block_a[i * block_n + j] = load_as_fp32(a, block_offset + i * N + j);
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
  

  // quantize and store back
  uint32_t round_helper = 1 << (23 - man_width - 1);
  for (int i = 0; i < block_m; i++) {
    for (int j = 0; j < block_n; j++) {
      if (pidx * block_m + i < M && pidy * block_n + j < N) {
        float out = single_quant(block_a[i * block_n + j], man_width, exp_width, scaling_factor, 0.0, round_helper);
        b[block_offset + i * N + j] = out;
      }
    }
  } 
}

torch::Tensor block_quant(
  torch::Tensor a,
  int man_width, int exp_width,
  int block_m, int block_n,
  bool is_stochastic_round
)
{
  CHECK_INPUT(a);
  cudaSetDevice(a.get_device());
  auto b = torch::zeros_like(a);
  uint32_t* rand_ptr = nullptr;
  if (is_stochastic_round) {
    torch::Tensor rand = torch::randint_like(a, INT_MAX, 
      torch::device(at::kCUDA).dtype(at::kUInt32));
    rand_ptr = rand.data_ptr<uint32_t>();
  }
  int M = a.size(0);
  int N = a.size(1);

  // // if (block_size < 32) {
    dim3 block(4, 8);
    dim3 grid(cdiv(M, block_m * 4), cdiv(N, block_n * 8));
    small_block_kernel<<<grid, block>>>(
      a.data_ptr<float>(), rand_ptr, b.data_ptr<float>(),
      block_m, block_n,
      M, N,
      man_width, exp_width,
      is_stochastic_round
    );
  // // } else {
    // dim3 block(block_m, block_n);
    // dim3 grid(cdiv(M, block_m), cdiv(N, block_n));
    // big_block_kernel<<<grid, block>>>(
    //   a.data_ptr<float>(), rand_ptr, b.data_ptr<float>(),
    //   block_m, block_n,
    //   M, N,
    //   man_width, exp_width,
    //   is_stochastic_round
    // );
  // }
  return b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_quant", &block_quant, "block quantization");
}
