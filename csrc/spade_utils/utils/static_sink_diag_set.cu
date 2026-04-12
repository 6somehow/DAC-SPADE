#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void static_sink_diag_set_kernel_impl(bool *mask,
                                                 const int *diag_mask_width,
                                                 int bsz, int num_heads,
                                                 int qNumBlocks, int kNumBlocks,
                                                 int sink_mask_width) {

  int batch_idx = blockIdx.x;
  int head_idx = blockIdx.y;

  if (batch_idx >= bsz || head_idx >= num_heads) {
    return;
  }

  int mask_offset =
      (batch_idx * num_heads + head_idx) * qNumBlocks * kNumBlocks;
  bool *current_mask = mask + mask_offset;

  int current_diag_width = diag_mask_width[batch_idx * num_heads + head_idx];

  int thread_id = threadIdx.x;
  int total_threads = blockDim.x;

  for (int q = thread_id; q < qNumBlocks; q += total_threads) {
    for (int k = 0; k < min(sink_mask_width, kNumBlocks); k++) {
      current_mask[q * kNumBlocks + k] = true;
    }
  }

  if (current_diag_width > 0) {
    for (int q = thread_id; q < qNumBlocks; q += total_threads) {
      float diag_center_k = (float)q * kNumBlocks / qNumBlocks;

      float half_width = (float)current_diag_width;
      int diag_start = max(0, (int)roundf(diag_center_k - half_width));
      int diag_end =
          min(kNumBlocks - 1, (int)roundf(diag_center_k + half_width));

      // 设置对角线区域为True
      for (int k = diag_start; k <= diag_end; k++) {
        current_mask[q * kNumBlocks + k] = true;
      }
    }
  }
}

torch::Tensor static_sink_diag_set(torch::Tensor mask,
                                 torch::Tensor diag_mask_width,
                                 int sink_mask_width) {

  TORCH_CHECK(mask.device().is_cuda(), "mask must be a CUDA tensor");
  TORCH_CHECK(diag_mask_width.device().is_cuda(),
              "diag_mask_width must be a CUDA tensor");
  TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be of type bool");
  TORCH_CHECK(diag_mask_width.dtype() == torch::kInt32,
              "diag_mask_width must be of type int32");

  auto sizes = mask.sizes();
  TORCH_CHECK(
      sizes.size() == 4,
      "mask must be 4-dimensional [bsz, num_heads, qNumBlocks, kNumBlocks]");

  int bsz = sizes[0];
  int num_heads = sizes[1];
  int qNumBlocks = sizes[2];
  int kNumBlocks = sizes[3];

  auto diag_sizes = diag_mask_width.sizes();
  TORCH_CHECK(diag_sizes.size() == 2,
              "diag_mask_width must be 2-dimensional [bsz, num_heads]");
  TORCH_CHECK(diag_sizes[0] == bsz && diag_sizes[1] == num_heads,
              "diag_mask_width shape must match [bsz, num_heads]");

  // 配置CUDA kernel启动参数
  dim3 grid(bsz, num_heads);
  int threads_per_block = std::min(1024, std::max(32, qNumBlocks));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // 启动CUDA kernel (原地操作)
  static_sink_diag_set_kernel_impl<<<grid, threads_per_block, 0, stream>>>(
      mask.data_ptr<bool>(), diag_mask_width.data_ptr<int>(), bsz, num_heads,
      qNumBlocks, kNumBlocks, sink_mask_width);

  // 检查CUDA错误
  cudaError_t err = cudaGetLastError();

  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel failed: ", cudaGetErrorString(err));
  return mask;
}
