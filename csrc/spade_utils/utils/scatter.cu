#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void scatter_kernel(bool *mask, const int *index,
                               const int *topk_size, int bsz, int num_heads,
                               int qNumBlocks, int kNumBlocks, int size) {
  // Each block processes one (batch, head, q_block) item.
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int q_block_idx = blockIdx.z;

  const int current_topk_val = topk_size[batch_idx * num_heads + head_idx];
  const int current_topk = min(current_topk_val, size);

  const int mask_base_offset =
      ((batch_idx * num_heads + head_idx) * qNumBlocks + q_block_idx) *
      kNumBlocks;
  const int index_base_offset =
      ((batch_idx * num_heads + head_idx) * qNumBlocks + q_block_idx) * size;

  for (int k = threadIdx.x; k < current_topk; k += blockDim.x) {
    const int target_k_idx = index[index_base_offset + k];

    if (target_k_idx >= 0 && target_k_idx < kNumBlocks) {
      mask[mask_base_offset + target_k_idx] = true;
    }
  }
}

void check_cuda_tensor(const torch::Tensor &tensor, const std::string &name) {
  CHECK_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
}

void check_tensor_shapes(const torch::Tensor &mask, const torch::Tensor &index,
                         const torch::Tensor &topk_size) {

  TORCH_CHECK(mask.dim() == 4, "mask must be 4-dimensional");
  TORCH_CHECK(index.dim() == 4, "index must be 4-dimensional");
  TORCH_CHECK(topk_size.dim() == 2, "topk_size must be 2-dimensional");

  int bsz = mask.size(0);
  int num_heads = mask.size(1);
  int qNumBlocks = mask.size(2);
  int kNumBlocks = mask.size(3);

  TORCH_CHECK(index.size(0) == bsz, "index batch size must match mask");
  TORCH_CHECK(index.size(1) == num_heads, "index num_heads must match mask");
  TORCH_CHECK(index.size(2) == qNumBlocks, "index qNumBlocks must match mask");

  TORCH_CHECK(topk_size.size(0) == bsz, "topk_size batch size must match mask");
  TORCH_CHECK(topk_size.size(1) == num_heads,
              "topk_size num_heads must match mask");
}

torch::Tensor scatter_cuda_forward(torch::Tensor mask, torch::Tensor index,
                                   torch::Tensor topk_size) {

  check_cuda_tensor(mask, "mask");
  check_cuda_tensor(index, "index");
  check_cuda_tensor(topk_size, "topk_size");

  TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool tensor");
  TORCH_CHECK(index.dtype() == torch::kInt32, "index must be int32 tensor");
  TORCH_CHECK(topk_size.dtype() == torch::kInt32,
              "topk_size must be int32 tensor");

  check_tensor_shapes(mask, index, topk_size);

  int bsz = mask.size(0);
  int num_heads = mask.size(1);
  int qNumBlocks = mask.size(2);
  int kNumBlocks = mask.size(3);
  int size = index.size(3);

  dim3 grid_size(bsz, num_heads, qNumBlocks);
  dim3 block_size(256);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  scatter_kernel<<<grid_size, block_size, 0, stream>>>(
      mask.data_ptr<bool>(), index.data_ptr<int>(), topk_size.data_ptr<int>(),
      bsz, num_heads, qNumBlocks, kNumBlocks, size);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel failed: ", cudaGetErrorString(err));

  return mask;
}

torch::Tensor scatter(torch::Tensor mask, torch::Tensor index,
                      torch::Tensor topk_size) {
  if (mask.device().is_cuda()) {
    return scatter_cuda_forward(mask, index, topk_size);
  } else {
    TORCH_CHECK(false, "scatter operation only supports CUDA tensors");
  }
}

