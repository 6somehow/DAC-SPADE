#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#define HEADDIM 128
#define WARP_SIZE 32
#define VECTOR_SIZE 4
#define BLOCK_SIZE (HEADDIM / VECTOR_SIZE)

#define ceil_div(a, b) ((a + b - 1) / b)

__device__ __forceinline__ float4 operator+(const float4 &a, const float4 &b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 operator*(const float4 &a, const float4 &b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ __forceinline__ float4 operator*(const float4 &a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__device__ __forceinline__ float4 operator/(const float4 &a, float b) {
  return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
__device__ __forceinline__ float4 &operator*=(float4 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
__device__ __forceinline__ float4 &operator+=(float4 &a, const float4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

__inline__ __device__ float warpAllReduceSum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return __shfl_sync(0xffffffff, val, 0);
}

template <typename T> struct TypeToFloat;

template <> struct TypeToFloat<__nv_bfloat16> {
  __device__ __forceinline__ static float convert(const __nv_bfloat16 &val) {
    return __bfloat162float(val);
  }
};

template <> struct TypeToFloat<__half> {
  __device__ __forceinline__ static float convert(const __half &val) {
    return __half2float(val);
  }
};

template <typename T>
__global__ __launch_bounds__(BLOCK_SIZE) void cossim_kernel_impl(
    const T *k, float *total_sim_sum, float *total_num_pairs,
    const int64_t frame_dim, const int64_t height_dim, const int64_t width_dim,
    const int64_t frame_block_size, const int64_t height_block_size,
    const int64_t width_block_size, const int num_heads,
    const int text_length) {
  int64_t bszIdx = blockIdx.x;
  int64_t headIdx = blockIdx.y;
  int64_t blockIdx_z = blockIdx.z;
  int64_t tid = threadIdx.x; // 0..31

  const int64_t num_frame_blocks = ceil_div(frame_dim, frame_block_size);
  const int64_t num_height_blocks = ceil_div(height_dim, height_block_size);
  const int64_t num_width_blocks = ceil_div(width_dim, width_block_size);

  const int64_t fhw_dim = frame_dim * height_dim * width_dim;
  const int64_t stride_k_fhw = num_heads * HEADDIM;
  const int64_t stride_k_bsz = (fhw_dim + text_length) * stride_k_fhw;

  int frameBlockidx = blockIdx_z / (num_height_blocks * num_width_blocks);
  int heightBlockIdx =
      (blockIdx_z % (num_height_blocks * num_width_blocks)) / num_width_blocks;
  int widthBlockIdx =
      (blockIdx_z % (num_height_blocks * num_width_blocks)) % num_width_blocks;

  float4 preNormSum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 cosSim4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  int numElements = 0;

  for (int fb = 0; fb < frame_block_size; fb++) {
    int64_t frameIdx = fb + frameBlockidx * frame_block_size;
    for (int hb = 0; hb < height_block_size; hb++) {
      int64_t heightIdx = hb + heightBlockIdx * height_block_size;
#pragma unroll(8)
      for (int w = 0; w < width_block_size; w++) {
        int64_t widthIdx = widthBlockIdx * width_block_size + w;

        if (frameIdx < frame_dim && heightIdx < height_dim &&
            widthIdx < width_dim) {
          numElements++;
          int64_t fhw_idx = frameIdx * height_dim * width_dim +
                            heightIdx * width_dim + widthIdx;
          int64_t kIdx_base = bszIdx * stride_k_bsz + fhw_idx * stride_k_fhw +
                              headIdx * HEADDIM;

          const T *k_ptr = k + kIdx_base + tid * VECTOR_SIZE;

          float4 fval4 = make_float4(TypeToFloat<T>::convert(k_ptr[0]),
                                     TypeToFloat<T>::convert(k_ptr[1]),
                                     TypeToFloat<T>::convert(k_ptr[2]),
                                     TypeToFloat<T>::convert(k_ptr[3]));

          float normFactor_scalar = (fval4.x * fval4.x) + (fval4.y * fval4.y) +
                                    (fval4.z * fval4.z) + (fval4.w * fval4.w);
          float total_norm_sq = warpAllReduceSum(normFactor_scalar);
          float norm = rsqrtf(total_norm_sq + 1e-8f);

          fval4 = fval4 * norm;

          cosSim4 += preNormSum4 * fval4 * 2.f;
          preNormSum4 += fval4;
        }
      }
    }
  }

  float cosSim_scalar = cosSim4.x + cosSim4.y + cosSim4.z + cosSim4.w;
  cosSim_scalar = warpAllReduceSum(cosSim_scalar);

  if (tid == 0) {
    int num_pairs = numElements * (numElements - 1);

    int64_t out_idx = bszIdx * num_heads + headIdx;

    atomicAdd(&total_sim_sum[out_idx], cosSim_scalar);
    atomicAdd(&total_num_pairs[out_idx], (float)num_pairs);
  }
}

at::Tensor cossim(at::Tensor k, std::vector<int64_t> seqlen3d,
                  std::vector<int64_t> block_shape,
                  std::vector<int64_t> num_blocks_on_axis,
                  int text_length = 0) {
  CHECK_INPUT(k);
  TORCH_CHECK(k.dtype() == torch::kBFloat16 || k.dtype() == torch::kFloat16,
              "k must be a bfloat16 or float16 tensor");

  const int bsz = k.size(0);
  const int num_heads = k.size(2);
  const int num_blocks_z =
      num_blocks_on_axis[0] * num_blocks_on_axis[1] * num_blocks_on_axis[2];

  auto options =
      torch::TensorOptions().device(k.device()).dtype(torch::kFloat32);
  auto total_sim_sum = torch::zeros({bsz, num_heads}, options);
  auto total_num_pairs = torch::zeros({bsz, num_heads}, options);

  dim3 grid(bsz, num_heads, num_blocks_z);
  dim3 block(BLOCK_SIZE);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  if (k.dtype() == torch::kBFloat16) {
    cossim_kernel_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16 *)k.data_ptr<at::BFloat16>(),
        total_sim_sum.data_ptr<float>(), total_num_pairs.data_ptr<float>(),
        seqlen3d[0], seqlen3d[1], seqlen3d[2], block_shape[0], block_shape[1],
        block_shape[2], num_heads, text_length);
  } else {
    cossim_kernel_impl<__half><<<grid, block, 0, stream>>>(
        (const __half *)k.data_ptr<at::Half>(), total_sim_sum.data_ptr<float>(),
        total_num_pairs.data_ptr<float>(), seqlen3d[0], seqlen3d[1],
        seqlen3d[2], block_shape[0], block_shape[1], block_shape[2], num_heads,
        text_length);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto avg_sim = total_sim_sum / (total_num_pairs + 1e-8);
  return avg_sim;
}
