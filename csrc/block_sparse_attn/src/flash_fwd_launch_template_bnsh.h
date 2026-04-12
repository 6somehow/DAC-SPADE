/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
/******************************************************************************
 * Adapted by Junxian Guo.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <c10/cuda/CUDAException.h>

#include "flash.h"
#include "flash_fwd_kernel_bnsh.h"
#include "hardware_info.h"
#include "static_switch.h"

namespace FLASH_NAMESPACE {

template <typename Kernel_traits, bool Is_causal, bool Is_local,
          bool Is_even_MN, bool Is_even_K, bool Is_exact_streaming>
__global__ void flash_fwd_block_bnsh_kernel(Flash_fwd_bnsh_params params) {
  static_assert(!(Is_causal && Is_local));
  FLASH_NAMESPACE::compute_block_attn_bnsh<
      Kernel_traits,
      /*Is_dropout=*/false,
      Is_causal,
      Is_local,
      /*Has_alibi=*/false,
      Is_even_MN,
      Is_even_K,
      /*Return_softmax=*/false,
      Is_exact_streaming>(params);
}

template <typename Kernel_traits, bool Is_causal>
void run_flash_fwd_block_bnsh(Flash_fwd_bnsh_params &params,
                              cudaStream_t stream) {
  constexpr size_t smem_size = Kernel_traits::kSmemSize;

  const int num_m_block =
      (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.b, params.h);
  const bool is_even_MN =
      params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
      params.seqlen_k % Kernel_traits::kBlockN == 0 &&
      params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;

  BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
      BOOL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) &&
                      !Is_causal,
                  Is_local, [&] {
        BOOL_SWITCH(params.is_exact_streaming, IsExactStreamingConst, [&] {
          auto kernel = &flash_fwd_block_bnsh_kernel<
              Kernel_traits, Is_causal, Is_local && !Is_causal,
              IsEvenMNConst &&
                  IsEvenKConst &&
                  !Is_local &&
                  Kernel_traits::kHeadDim <= 128,
              IsEvenKConst, IsExactStreamingConst && Is_causal>;
          if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size));
          }
          kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
      });
    });
  });
}

template <typename T, bool Is_causal>
void run_mha_fwd_block_bnsh_hdim32(Flash_fwd_bnsh_params &params,
                                   cudaStream_t stream) {
  constexpr static int Headdim = 32;
  if (params.m_block_dim % 128 != 0 || params.n_block_dim % 128 != 0) {
    run_flash_fwd_block_bnsh<
        Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
        Is_causal>(params, stream);
    return;
  }
  run_flash_fwd_block_bnsh<
      Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>,
      Is_causal>(params, stream);
}

template <typename T, bool Is_causal>
void run_mha_fwd_block_bnsh_hdim64(Flash_fwd_bnsh_params &params,
                                   cudaStream_t stream) {
  constexpr static int Headdim = 64;
  if (params.m_block_dim % 128 != 0 || params.n_block_dim % 128 != 0) {
    run_flash_fwd_block_bnsh<
        Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
        Is_causal>(params, stream);
    return;
  }
  run_flash_fwd_block_bnsh<
      Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>,
      Is_causal>(params, stream);
}

template <typename T, bool Is_causal>
void run_mha_fwd_block_bnsh_hdim128(Flash_fwd_bnsh_params &params,
                                    cudaStream_t stream) {
  constexpr static int Headdim = 128;
  if (params.m_block_dim % 128 != 0 || params.n_block_dim % 128 != 0) {
    run_flash_fwd_block_bnsh<
        Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
        Is_causal>(params, stream);
    return;
  }
  auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
  bool is_sm8x = cc_major == 8 && cc_minor > 0;
  if (is_sm8x) {
    if constexpr (!Is_causal) {
      run_flash_fwd_block_bnsh<
          Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>,
          Is_causal>(params, stream);
    } else {
      run_flash_fwd_block_bnsh<
          Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
          Is_causal>(params, stream);
    }
  } else {
    run_flash_fwd_block_bnsh<
        Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>,
        Is_causal>(params, stream);
  }
}

} // namespace FLASH_NAMESPACE
