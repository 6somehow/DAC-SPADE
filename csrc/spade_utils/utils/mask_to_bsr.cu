#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA 内核，功能与 Triton 内核相同
__global__ void
mask_to_bsr_kernel(const bool *__restrict__ mask_ptr, int *__restrict__ bsr_ptr,
                   int *__restrict__ bsr_num_ptr,
                   // mask_ptr 的步长
                   const int64_t mask_bs_stride, const int64_t mask_h_stride,
                   const int64_t mask_q_stride, const int64_t mask_kv_stride,
                   // bsr_ptr 的步长
                   const int64_t bsr_bs_stride, const int64_t bsr_h_stride,
                   const int64_t bsr_q_stride, const int64_t bsr_kv_stride,
                   // bsr_num_ptr 的步长
                   const int64_t bsr_num_bs_stride,
                   const int64_t bsr_num_h_stride,
                   const int64_t bsr_num_q_stride,
                   // 维度信息
                   const int bsz, const int num_heads, const int q_num_blocks,
                   const int num_kv_blocks) {
  // 将 CUDA 网格索引映射到逻辑维度 (b, h, q)
  // Triton 的 program_id(0) -> b -> CUDA 的 blockIdx.z
  // Triton 的 program_id(1) -> h -> CUDA 的 blockIdx.y
  // Triton 的 program_id(2) -> q -> CUDA 的 blockIdx.x
  const int q = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  // 边界检查，防止越界访问
  if (b >= bsz || h >= num_heads || q >= q_num_blocks) {
    return;
  }

  // 计算当前 (b, h, q) 对应的基地址
  const bool *mask_ptr_base =
      mask_ptr + b * mask_bs_stride + h * mask_h_stride + q * mask_q_stride;
  int *bsr_ptr_base =
      bsr_ptr + b * bsr_bs_stride + h * bsr_h_stride + q * bsr_q_stride;

  int num = 0;
  // 遍历 kv_blocks 维度
  for (int i = 0; i < num_kv_blocks; ++i) {
    // 读取 mask 值
    if (mask_ptr_base[i * mask_kv_stride]) {
      // 如果 mask 为 true，则存储其索引并增加计数器
      bsr_ptr_base[num * bsr_kv_stride] = i;
      num++;
    }
  }

  // 存储该行非零块的总数
  int *bsr_num_ptr_target = bsr_num_ptr + b * bsr_num_bs_stride +
                            h * bsr_num_h_stride + q * bsr_num_q_stride;
  *bsr_num_ptr_target = num;
}

// C++ 封装函数，用于从 Python 调用
void mask_to_bsr(const torch::Tensor &sparse_mask, torch::Tensor &bsr,
                         torch::Tensor &num_blocks) {
  // 检查张量是否在 CUDA 上并且是连续的
  TORCH_CHECK(sparse_mask.device().is_cuda(), "sparse_mask must be a CUDA tensor");
  TORCH_CHECK(bsr.device().is_cuda(), "bsr must be a CUDA tensor");
  TORCH_CHECK(num_blocks.device().is_cuda(), "num_blocks must be a CUDA tensor");
  TORCH_CHECK(sparse_mask.is_cuda(), "sparse_mask must be a CUDA tensor");
  TORCH_CHECK(bsr.is_cuda(), "bsr must be a CUDA tensor");
  TORCH_CHECK(num_blocks.is_cuda(), "num_blocks must be a CUDA tensor");
  TORCH_CHECK(sparse_mask.scalar_type() == torch::kBool,
              "sparse_mask must be a boolean tensor");
  TORCH_CHECK(bsr.scalar_type() == torch::kInt32,
              "bsr must be an int32 tensor");
  TORCH_CHECK(num_blocks.scalar_type() == torch::kInt32,
              "num_blocks must be an int32 tensor");

  // 获取维度信息
  const int bsz = sparse_mask.size(0);
  const int num_heads = sparse_mask.size(1);
  const int q_num_blocks = sparse_mask.size(2);
  const int k_num_blocks = sparse_mask.size(3);

  // 获取数据指针
  const bool *mask_ptr = sparse_mask.data_ptr<bool>();
  int *bsr_ptr = bsr.data_ptr<int>();
  int *bsr_num_ptr = num_blocks.data_ptr<int>();

  // 定义 CUDA 网格和块的维度
  const dim3 grid(q_num_blocks, num_heads, bsz);
  const dim3 block(1); // 每个块一个线程
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // 启动 CUDA 内核
  mask_to_bsr_kernel<<<grid, block, 0, stream>>>(
      mask_ptr, bsr_ptr, bsr_num_ptr, sparse_mask.stride(0),
      sparse_mask.stride(1), sparse_mask.stride(2), sparse_mask.stride(3),
      bsr.stride(0), bsr.stride(1), bsr.stride(2), bsr.stride(3),
      num_blocks.stride(0), num_blocks.stride(1), num_blocks.stride(2), bsz,
      num_heads, q_num_blocks, k_num_blocks);

  // 检查 CUDA 错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}
