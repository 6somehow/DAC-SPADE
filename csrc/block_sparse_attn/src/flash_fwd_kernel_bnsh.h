/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
/******************************************************************************
 * Adapted by Junxian Guo.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include "flash_fwd_kernel.h"

namespace FLASH_NAMESPACE {

template <typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local,
          bool Has_alibi, bool Is_even_MN, bool Is_even_K,
          bool Return_softmax, bool Is_exact_streaming, typename Params>
inline __device__ void compute_block_attn_bnsh(const Params &params) {
  FLASH_NAMESPACE::compute_block_attn<
      Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN,
      Is_even_K, Return_softmax, Is_exact_streaming>(params);
}

} // namespace FLASH_NAMESPACE
