// modify from https://github.com/hao-ai-lab/FastVideo/blob/c591d6d2a673e717c0ddc2adb8fbaeb3db57c3df/fastvideo-kernel/csrc/attention/block_sparse_h100.cu

#include <torch/extension.h>
<<<<<<< HEAD
=======
#if defined(KITTENS_HOPPER) && !defined(KITTENS_BLACKWELL) && __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
namespace kittens {
using fp4_2 = __nv_fp4x2_e2m1;
using fp4_4 = __nv_fp4x4_e2m1;
}  // namespace kittens
#endif
>>>>>>> dev
#include "kittens.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <iostream>

#define CHECK_CUDA_ERROR(error) TORCH_CHECK((error) == cudaSuccess, cudaGetErrorString(error))

using namespace kittens;
namespace cg = cooperative_groups;
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
template <int D>
struct fwd_attend_ker_tile_dims
{
};
template <>
struct fwd_attend_ker_tile_dims<64>
{
  constexpr static int tile_width = (64);
  constexpr static int qo_height = (4 * 16);
  constexpr static int kv_height = (4 * 16);
};
template <>
struct fwd_attend_ker_tile_dims<128>
{
  constexpr static int tile_width = (128);
  constexpr static int qo_height = (4 * 16);
  constexpr static int kv_height = (4 * 16);
};
template <int D>
struct fwd_globals
{
  using q_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height,
                       fwd_attend_ker_tile_dims<D>::tile_width>;
  using k_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height,
                       fwd_attend_ker_tile_dims<D>::tile_width>;
  using v_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height,
                       fwd_attend_ker_tile_dims<D>::tile_width>;
  using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height,
                                  fwd_attend_ker_tile_dims<D>::tile_width>>;
  using o_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height,
                       fwd_attend_ker_tile_dims<D>::tile_width>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
  using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
  using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;

  q_gl q;
  k_gl k;
  v_gl v;
  l_gl l;
  o_gl o;

  const int N;
  const int hr;
  const int max_kv_blocks_per_q;

  int32_t *__restrict__ q2k_block_sparse_index;
  int32_t *__restrict__ q2k_block_sparse_num;
};

template <int D>
__global__ __launch_bounds__(128, 4) void fwd_attend_ker(
    const __grid_constant__ fwd_globals<D> g)
{ // use block size of 64
  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);

  using K = fwd_attend_ker_tile_dims<D>;

  using q_tile = st_bf<64, K::tile_width>;
  using k_tile = st_bf<64, K::tile_width>;
  using v_tile = st_bf<64, K::tile_width>;
  using l_col_vec = col_vec<st_fl<64, K::tile_width>>;
  using o_tile = st_bf<64, K::tile_width>;

  q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();

  k_tile(&k_smem)[1] = al.allocate<k_tile, 1>();

  v_tile(&v_smem)[1] = al.allocate<v_tile, 1>();

  l_col_vec(&l_smem)[1] = al.allocate<l_col_vec, 1>();

  auto(*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

  int kv_head_idx = blockIdx.y / g.hr;
  int seq_idx = blockIdx.x;

  int32_t *q2k_block_sparse_index_ptr =
      g.q2k_block_sparse_index +
      blockIdx.z * gridDim.y * gridDim.x * g.max_kv_blocks_per_q +
      blockIdx.y * gridDim.x * g.max_kv_blocks_per_q +
      blockIdx.x * g.max_kv_blocks_per_q;
  int32_t *q2k_block_sparse_num_ptr = g.q2k_block_sparse_num +
                                      blockIdx.z * gridDim.y * gridDim.x +
                                      blockIdx.y * gridDim.x + blockIdx.x;
  int32_t kv_blocks = q2k_block_sparse_num_ptr[0];
  __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived, v_smem_arrived;
  if (threadIdx.x == 0)
  {
    int32_t kv_block_index = q2k_block_sparse_index_ptr[0];

    init_semaphore(qsmem_semaphore, 0, 1);
    init_semaphore(k_smem_arrived, 0, 1);
    init_semaphore(v_smem_arrived, 0, 1);

    // preload q block
    coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
    tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
    tma::load_async(q_smem[0], g.q, q_tile_idx, qsmem_semaphore);

    // preload the zeroth block of kv
    tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
    coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
    tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);

    tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
    coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
    tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
  }
  __syncthreads();

  rt_fl<16, 64> att_block;
  rt_bf<16, 64> att_block_mma;
  rt_fl<16, K::tile_width> o_reg;

  col_vec<rt_fl<16, 64>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

<<<<<<< HEAD
  neg_infty(max_vec);
  zero(norm_vec);
  zero(o_reg);
=======
  warp::neg_infty(max_vec);
  warp::zero(norm_vec);
  warp::zero(o_reg);
>>>>>>> dev

  // wait for q block
  wait(qsmem_semaphore, 0);

  for (int kv_idx = 0; kv_idx < kv_blocks - 1; kv_idx++)
  {
    // preload kv index
    int32_t kv_block_index = q2k_block_sparse_index_ptr[kv_idx + 1];

    // wait k
    wait(k_smem_arrived, kv_idx % 2);

    // compute QK^T
    warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);

<<<<<<< HEAD
    copy(max_vec_last_scaled, max_vec);
    if constexpr (D == 64)
    {
      mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
    }
    else
    {
      mul(max_vec_last_scaled, max_vec_last_scaled,
          1.44269504089f * 0.08838834764f);
=======
    warp::copy(max_vec_last_scaled, max_vec);
    if constexpr (D == 64)
    {
      warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
    }
    else
    {
      warp::mul(max_vec_last_scaled, max_vec_last_scaled,
                1.44269504089f * 0.08838834764f);
>>>>>>> dev
    }

    warpgroup::mma_async_wait();

    // load K
    if (threadIdx.x == 0)
    {
      tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
      coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
      tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);
    }

<<<<<<< HEAD
    row_max(max_vec, att_block, max_vec);

    if constexpr (D == 64)
    {
      mul(att_block, att_block, 1.44269504089f * 0.125f);
      mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
    }
    else
    {
      mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
      mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
    }

    sub_row(att_block, att_block, max_vec_scaled);
    exp2(att_block, att_block);
    sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    exp2(max_vec_last_scaled, max_vec_last_scaled);
    mul(norm_vec, norm_vec, max_vec_last_scaled);
    row_sum(norm_vec, att_block, norm_vec);
    add(att_block, att_block, 0.f);
    copy(att_block_mma, att_block);
    mul_row(o_reg, o_reg, max_vec_last_scaled);
=======
    warp::row_max(max_vec, att_block, max_vec);

    if constexpr (D == 64)
    {
      warp::mul(att_block, att_block, 1.44269504089f * 0.125f);
      warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
    }
    else
    {
      warp::mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
      warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
    }

    warp::sub_row(att_block, att_block, max_vec_scaled);
    warp::exp2(att_block, att_block);
    warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
    warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
    warp::row_sum(norm_vec, att_block, norm_vec);
    warp::add(att_block, att_block, 0.f);
    warp::copy(att_block_mma, att_block);
    warp::mul_row(o_reg, o_reg, max_vec_last_scaled);
>>>>>>> dev

    // wait v
    wait(v_smem_arrived, kv_idx % 2);

    // compute SV
    warpgroup::mma_AB(o_reg, att_block_mma, v_smem[0]);
    warpgroup::mma_async_wait();

    // load V
    if (threadIdx.x == 0)
    {
      tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
      coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
      tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
    }
  }

  // last iter
  {
    int kv_idx = kv_blocks - 1;
    // wait k
    wait(k_smem_arrived, kv_idx % 2);

    // compute QK^T
    warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);

<<<<<<< HEAD
    copy(max_vec_last_scaled, max_vec);
    if constexpr (D == 64)
    {
      mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
    }
    else
    {
      mul(max_vec_last_scaled, max_vec_last_scaled,
          1.44269504089f * 0.08838834764f);
=======
    warp::copy(max_vec_last_scaled, max_vec);
    if constexpr (D == 64)
    {
      warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
    }
    else
    {
      warp::mul(max_vec_last_scaled, max_vec_last_scaled,
                1.44269504089f * 0.08838834764f);
>>>>>>> dev
    }

    warpgroup::mma_async_wait();

<<<<<<< HEAD
    row_max(max_vec, att_block, max_vec);

    if constexpr (D == 64)
    {
      mul(att_block, att_block, 1.44269504089f * 0.125f);
      mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
    }
    else
    {
      mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
      mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
    }

    sub_row(att_block, att_block, max_vec_scaled);
    exp2(att_block, att_block);
    sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    exp2(max_vec_last_scaled, max_vec_last_scaled);
    mul(norm_vec, norm_vec, max_vec_last_scaled);
    row_sum(norm_vec, att_block, norm_vec);
    add(att_block, att_block, 0.f);
    copy(att_block_mma, att_block);
    mul_row(o_reg, o_reg, max_vec_last_scaled);
=======
    warp::row_max(max_vec, att_block, max_vec);

    if constexpr (D == 64)
    {
      warp::mul(att_block, att_block, 1.44269504089f * 0.125f);
      warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
    }
    else
    {
      warp::mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
      warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
    }

    warp::sub_row(att_block, att_block, max_vec_scaled);
    warp::exp2(att_block, att_block);
    warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
    warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
    warp::row_sum(norm_vec, att_block, norm_vec);
    warp::add(att_block, att_block, 0.f);
    warp::copy(att_block_mma, att_block);
    warp::mul_row(o_reg, o_reg, max_vec_last_scaled);
>>>>>>> dev

    // wait v
    wait(v_smem_arrived, kv_idx % 2);

    // compute SV
    warpgroup::mma_AB(o_reg, att_block_mma, v_smem[0]);
    warpgroup::mma_async_wait();
  }

<<<<<<< HEAD
  div_row(o_reg, o_reg, norm_vec);
=======
  warp::div_row(o_reg, o_reg, norm_vec);
>>>>>>> dev
  warpgroup::store(o_smem[0], o_reg);
  __syncthreads();

  // TK store_async internally calls syncwarp so we need to route on warp level
  if (threadIdx.x / 32 == 0)
  {
    coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
    tma::store_async(g.o, o_smem[0], o_tile_idx);
  }

<<<<<<< HEAD
  mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
  log(norm_vec, norm_vec);
  add(norm_vec, norm_vec, max_vec_scaled);

  if constexpr (D == 64)
  {
    mul(norm_vec, norm_vec, -8.0f);
  }
  else
  {
    mul(norm_vec, norm_vec, -11.313708499f);
=======
  warp::mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
  warp::log(norm_vec, norm_vec);
  warp::add(norm_vec, norm_vec, max_vec_scaled);

  if constexpr (D == 64)
  {
    warp::mul(norm_vec, norm_vec, -8.0f);
  }
  else
  {
    warp::mul(norm_vec, norm_vec, -11.313708499f);
>>>>>>> dev
  }

  warpgroup::store(l_smem[0], norm_vec);
  __syncthreads();

  if (threadIdx.x / 32 == 0)
  {
    coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, seq_idx};
    tma::store_async(g.l, l_smem[0], tile_idx);
  }
  tma::store_async_wait();
}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------- Backward preparation kernel
// -----------------------------------
// ---------------------------------------------------------------------------------------------------

template <int D>
struct bwd_prep_globals
{
  using og_tile = st_bf<4 * 16, D>;
  using o_tile = st_bf<4 * 16, D>;
  using d_tile = col_vec<st_fl<4 * 16, D>>;

  using og_gl = gl<bf16, -1, -1, -1, -1, og_tile>;
  using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;
  using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

  og_gl og;
  o_gl o;
  d_gl d;
};

constexpr int PREP_NUM_WARPS = (1);
template <int D>
__global__ __launch_bounds__(
    PREP_NUM_WARPS *kittens::WARP_THREADS,
    (D == 64)
        ? 6 / PREP_NUM_WARPS
        : 3 / PREP_NUM_WARPS) void bwd_attend_prep_ker(const __grid_constant__
                                                           bwd_prep_globals<D>
                                                               g)
{
  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);

  int warpid = kittens::warpid();

  using og_tile = st_bf<4 * 16, D>;
  using o_tile = st_bf<4 * 16, D>;
  using d_tile = col_vec<st_fl<4 * 16, D>>;

  og_tile(&og_smem)[PREP_NUM_WARPS] = al.allocate<og_tile, PREP_NUM_WARPS>();
  o_tile(&o_smem)[PREP_NUM_WARPS] = al.allocate<o_tile, PREP_NUM_WARPS>();
  d_tile(&d_smem)[PREP_NUM_WARPS] = al.allocate<d_tile, PREP_NUM_WARPS>();

  rt_fl<4 * 16, D> og_reg, o_reg;
  col_vec<rt_fl<4 * 16, D>> d_reg;

  __shared__ kittens::semaphore smem_semaphore;

  if (threadIdx.x == 0)
  {
    init_semaphore(smem_semaphore, 0, 1);
    tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * PREP_NUM_WARPS * 2);
  }
  __syncthreads();

  if (warpid == 0)
  {
    for (int w = 0; w < PREP_NUM_WARPS; w++)
    {
      coord<o_tile> tile_idx = {blockIdx.z, blockIdx.y,
                                (blockIdx.x * PREP_NUM_WARPS) + w, 0};
      tma::load_async(o_smem[w], g.o, tile_idx, smem_semaphore);
      tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
    }
  }

  wait(smem_semaphore, 0);
<<<<<<< HEAD
  load(o_reg, o_smem[warpid]);
  load(og_reg, og_smem[warpid]);
  mul(og_reg, og_reg, o_reg);
  row_sum(d_reg, og_reg);
  store(d_smem[warpid], d_reg);
=======
  warp::load(o_reg, o_smem[warpid]);
  warp::load(og_reg, og_smem[warpid]);
  warp::mul(og_reg, og_reg, o_reg);
  warp::row_sum(d_reg, og_reg);
  warp::store(d_smem[warpid], d_reg);
>>>>>>> dev
  __syncthreads();

  if (warpid == 0)
  {
    for (int w = 0; w < PREP_NUM_WARPS; w++)
    {
      coord<d_tile> tile_idx = {blockIdx.z, blockIdx.y, 0,
                                (blockIdx.x * PREP_NUM_WARPS) + w};
      tma::store_async(g.d, d_smem[w], tile_idx);
    }
  }
  tma::store_async_wait();
}

template <int D>
struct bwd_attend_ker_tile_dims
{
};
template <>
struct bwd_attend_ker_tile_dims<64>
{
  constexpr static int tile_width = (64);
  constexpr static int tile_h = (4 * 16);
  constexpr static int tile_h_qo = (4 * 16);
};
template <>
struct bwd_attend_ker_tile_dims<128>
{
  constexpr static int tile_width = (128);
  constexpr static int tile_h = (4 * 16);
  constexpr static int tile_h_qo = (4 * 16);
};

__device__ static inline void stream_tile(auto &reg_tile, auto &smem_vec,
                                          int tic)
{
#pragma unroll
  for (int i = 0; i < 4; i++)
  {
    int base_col = 16 * i + 2 * (kittens::laneid() % 4);
    reg_tile.tiles[0][i].data[0] = *(float2 *)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][i].data[1] = *(float2 *)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][i].data[2] = *(float2 *)&smem_vec[tic][base_col + 8];
    reg_tile.tiles[0][i].data[3] = *(float2 *)&smem_vec[tic][base_col + 8];
  }
}

__device__ static inline void stream_sub_tile(auto &reg_tile, auto &smem_vec,
                                              int tic)
{
#pragma unroll
  for (int i = 0; i < 4; i++)
  {
    int base_col = 16 * i + 2 * (laneid() % 4);
    reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(
        reg_tile.tiles[0][i].data[0], *(float2 *)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(
        reg_tile.tiles[0][i].data[1], *(float2 *)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(
        reg_tile.tiles[0][i].data[2], *(float2 *)&smem_vec[tic][base_col + 8]);
    reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(
        reg_tile.tiles[0][i].data[3], *(float2 *)&smem_vec[tic][base_col + 8]);
  }
}

#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

std::vector<torch::Tensor> tk_block_sparse_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor q2k_block_sparse_index, torch::Tensor q2k_block_sparse_num)
{
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  auto batch = q.size(0);
  auto seq_len = q.size(2);
  auto head_dim = q.size(3);
  auto qo_heads = q.size(1);
  auto kv_heads = k.size(1);
  auto max_kv_blocks_per_q = q2k_block_sparse_index.size(3);
  auto num_q_blocks = q2k_block_sparse_index.size(2);
  TORCH_CHECK(
      batch == 1,
      "Batch size dim will be removed in the future, please set batch to 1");
  TORCH_CHECK(num_q_blocks * 64 == seq_len,
              "This kernel supports variable block size, but it assumes the "
              "input sequence is properly padded.");
  TORCH_CHECK(num_q_blocks == q2k_block_sparse_index.size(2),
              "Number of Q blocks does not match between "
              "q2k_block_sparse_index and block_size");
  // check to see that these dimensions match for all inputs
  TORCH_CHECK(q.size(0) == batch,
              "Q batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(k.size(0) == batch,
              "K batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(v.size(0) == batch,
              "V batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(q2k_block_sparse_index.size(0) == batch,
              "q2k_block_sparse_index batch dimension - idx 0 - must match for "
              "all inputs");
  TORCH_CHECK(q2k_block_sparse_num.size(0) == batch,
              "q2k_block_sparse_num batch dimension - idx 0 - must match for "
              "all inputs");

  TORCH_CHECK(
      q.size(2) == seq_len,
      "Q sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      k.size(2) == seq_len,
      "K sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      v.size(2) == seq_len,
      "V sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(q2k_block_sparse_index.size(2) == seq_len / BLOCK_M,
              "q2k_block_sparse_index idx 2 - must match seq_len / BLOCK_M");
  TORCH_CHECK(q2k_block_sparse_num.size(2) == seq_len / BLOCK_M,
              "q2k_block_sparse_num idx 2 - must match seq_len / BLOCK_M");

  TORCH_CHECK(
      q.size(3) == head_dim,
      "Q head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      k.size(3) == head_dim,
      "K head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      v.size(3) == head_dim,
      "V head dimension - idx 3 - must match for all non-vector inputs");

  TORCH_CHECK(qo_heads >= kv_heads,
              "QO heads must be greater than or equal to KV heads");
  TORCH_CHECK(qo_heads % kv_heads == 0,
              "QO heads must be divisible by KV heads");
  TORCH_CHECK(q.size(1) == qo_heads,
              "QO head dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(k.size(1) == kv_heads,
              "KV head dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(v.size(1) == kv_heads,
              "KV head dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(q2k_block_sparse_index.size(1) == qo_heads,
              "q2k_block_sparse_index head dimension - idx 1 - must match for "
              "all inputs");
  TORCH_CHECK(q2k_block_sparse_num.size(1) == qo_heads,
              "q2k_block_sparse_num head dimension - idx 1 - must match for "
              "all inputs");
  auto hr = qo_heads / kv_heads;

  c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
  c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
  c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();

  bf16 *d_q = reinterpret_cast<bf16 *>(q_ptr);
  bf16 *d_k = reinterpret_cast<bf16 *>(k_ptr);
  bf16 *d_v = reinterpret_cast<bf16 *>(v_ptr);

  // for the returned outputs
  torch::Tensor o = torch::empty(
      {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(head_dim)},
      v.options());

  torch::Tensor l_vec = torch::empty(
      {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(1)},
      torch::TensorOptions()
          .dtype(torch::kFloat)
          .device(q.device())
          .memory_format(at::MemoryFormat::Contiguous));

  bf16 *o_ptr = reinterpret_cast<bf16 *>(o.data_ptr<c10::BFloat16>());
  bf16 *d_o = reinterpret_cast<bf16 *>(o_ptr);

  float *l_ptr = reinterpret_cast<float *>(l_vec.data_ptr<float>());
  float *d_l = reinterpret_cast<float *>(l_ptr);

  // cudadevicesynchronize();
  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (head_dim == 64)
  {
    using q_tile = st_bf<fwd_attend_ker_tile_dims<64>::qo_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims<64>::kv_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims<64>::kv_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<64>::qo_height,
                                    fwd_attend_ker_tile_dims<64>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims<64>::qo_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;

    using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = fwd_globals<64>;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads), 1U,
                    static_cast<unsigned int>(seq_len)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 64U};

    globals g{qg_arg,
              kg_arg,
              vg_arg,
              lg_arg,
              og_arg,
              static_cast<int>(seq_len),
              static_cast<int>(hr),
              static_cast<int>(max_kv_blocks_per_q),
              reinterpret_cast<int32_t *>(q2k_block_sparse_index.data_ptr()),
              reinterpret_cast<int32_t *>(q2k_block_sparse_num.data_ptr())};

    constexpr int mem_size = 54000;

    dim3 grid(seq_len / (64), qo_heads, batch);

    cudaFuncSetAttribute(fwd_attend_ker<64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    fwd_attend_ker<64><<<grid, (128), mem_size, stream>>>(g);

    CHECK_CUDA_ERROR(cudaGetLastError());
    // cudaStreamSynchronize(stream);
  }

  if (head_dim == 128)
  {
    using q_tile = st_bf<fwd_attend_ker_tile_dims<128>::qo_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims<128>::kv_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims<128>::kv_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height,
                                    fwd_attend_ker_tile_dims<128>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims<128>::qo_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;

    using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = fwd_globals<128>;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads), 1U,
                    static_cast<unsigned int>(seq_len)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 128U};

    globals g{qg_arg,
              kg_arg,
              vg_arg,
              lg_arg,
              og_arg,
              static_cast<int>(seq_len),
              static_cast<int>(hr),
              static_cast<int>(max_kv_blocks_per_q),
              reinterpret_cast<int32_t *>(q2k_block_sparse_index.data_ptr()),
              reinterpret_cast<int32_t *>(q2k_block_sparse_num.data_ptr())};

    constexpr int mem_size = 54000;

    dim3 grid(seq_len / (64), qo_heads, batch);

    cudaFuncSetAttribute(fwd_attend_ker<128>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    fwd_attend_ker<128><<<grid, (128), mem_size, stream>>>(g);

    CHECK_CUDA_ERROR(cudaGetLastError());
    // cudaStreamSynchronize(stream);
  }

  return {o, l_vec};
  // cudadevicesynchronize();
}
