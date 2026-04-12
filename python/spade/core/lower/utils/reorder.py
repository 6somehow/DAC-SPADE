from math import prod

import torch
import triton
import triton.language as tl


@triton.jit
def reorder_kernel(
    k,  #[bsz,seqlen,numHead,headDim]
    new_k,  #[bsz,numBlock,blockSize,numListLength,headDim]
    headIndices,  #[numListLength]
    padding_val: tl.constexpr,
    stride_k0: tl.constexpr,
    stride_k1: tl.constexpr,
    stride_k2: tl.constexpr,
    stride_k3: tl.constexpr,
    stride_new_k0,
    stride_new_k1,
    stride_new_k2,
    stride_new_k3,
    stride_new_k4,
    frameDim: tl.constexpr,
    heightDim: tl.constexpr,
    widthDim: tl.constexpr,
    frameBlockSize: tl.constexpr,
    heightBlockSize: tl.constexpr,
    widthBlockSize: tl.constexpr,
    numFrameBlock: tl.constexpr,
    numHeightBlock: tl.constexpr,
    numWidthBlock: tl.constexpr,
    numHeads: tl.constexpr,
    headDim: tl.constexpr,
):
    DTYPE: tl.constexpr = k.dtype.element_ty
    bszIdx = tl.program_id(0).to(tl.int64)
    headListIdx = tl.program_id(1).to(tl.int64)
    headIdx = tl.load(headIndices + headListIdx).to(tl.int64)
    blockIdx = tl.program_id(2).to(tl.int64)
    frameBlockidx = blockIdx // (numHeightBlock * numWidthBlock)
    heightBlockIdx = (blockIdx %
                      (numHeightBlock * numWidthBlock)) // numWidthBlock
    widthBlockIdx = (blockIdx %
                     (numHeightBlock * numWidthBlock)) % numWidthBlock
    tid = tl.arange(0, headDim)
    VECSIZE: tl.constexpr = triton.next_power_of_2(widthBlockSize)
    local_width_indices = tl.arange(0, VECSIZE).to(tl.int64)
    for fb in tl.static_range(frameBlockSize):
        frameIdx = fb + frameBlockidx * frameBlockSize
        for hb in tl.static_range(heightBlockSize):
            heightIdx = hb + heightBlockIdx * heightBlockSize
            widthIdx = widthBlockIdx * widthBlockSize + local_width_indices
            valid_width = local_width_indices < widthBlockSize
            widthIdxMask = valid_width & (widthIdx < widthDim)

            kIdx = (bszIdx * stride_k0 +
                    (frameIdx * heightDim * widthDim + heightIdx * widthDim +
                     widthIdx[:, None]) * stride_k1 + headIdx * stride_k2 +
                    tid[None, :] * stride_k3)
            if frameIdx < frameDim and heightIdx < heightDim:
                kBlock = tl.load(k + kIdx,
                                 mask=widthIdxMask[:, None],
                                 other=padding_val)
            else:
                kBlock = tl.full((VECSIZE, headDim), padding_val, dtype=DTYPE)

            bblockIdx = fb * heightBlockSize * widthBlockSize + \
                        hb * widthBlockSize + \
                        local_width_indices[:, None]
            bblockIdx = tl.where(valid_width[:, None], bblockIdx, 0)

            new_kIdx = (bszIdx * stride_new_k0 + blockIdx * stride_new_k1 +
                        bblockIdx * stride_new_k2 +
                        headListIdx * stride_new_k3 +
                        tid[None, :] * stride_new_k4)

            tl.store(new_k + new_kIdx,
                     kBlock,
                     mask=(local_width_indices < widthBlockSize)[:, None])


def ReorderTensorBSNH(k_new: torch.Tensor, k: torch.Tensor,
                      headIndices: torch.Tensor, padding_val: float,
                      seqlenDim3, blocksSize, numBlocks):
    bsz, seqlen, numHeads, headDim = k.size()
    headIndices = headIndices.to(k.device).contiguous()
    numHeadList = headIndices.numel()
    assert seqlen == prod(seqlenDim3), f"seqlen {seqlen} != {prod(seqlenDim3)}"
    numBlock = prod(numBlocks)
    grid = (bsz, numHeadList, numBlock)

    reorder_kernel[grid](
        k,
        k_new,
        headIndices,
        padding_val,
        *k.stride(),
        *k_new.stride(),
        *seqlenDim3,
        *blocksSize,
        *numBlocks,
        numHeads,
        headDim,
    )
    return k_new


@triton.jit
def reorder_back_kernel(
    o,  # [bsz,numBlock,blockSize,numHeadList,headDim]
    new_o,  # [bsz,seqlen,num_heads,headDim]
    headIndices,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    stride_new_o0: tl.constexpr,
    stride_new_o1: tl.constexpr,
    stride_new_o2: tl.constexpr,
    stride_new_o3: tl.constexpr,
    frameDim: tl.constexpr,
    frameBlockSize: tl.constexpr,
    numFrameBlock: tl.constexpr,
    heightDim: tl.constexpr,
    heightBlockSize: tl.constexpr,
    numHeightBlock: tl.constexpr,
    widthDim: tl.constexpr,
    widthBlockSize: tl.constexpr,
    numWidthBlock: tl.constexpr,
    numHeads: tl.constexpr,
    headDim: tl.constexpr,
):
    bszIdx = tl.program_id(0).to(tl.int64)
    headListIdx = tl.program_id(1).to(tl.int64)
    headIdx = tl.load(headIndices + headListIdx).to(tl.int64)
    blockIdx = tl.program_id(2).to(tl.int64)
    frameBlockidx = blockIdx // (numHeightBlock * numWidthBlock)
    heightBlockIdx = (blockIdx %
                      (numHeightBlock * numWidthBlock)) // numWidthBlock
    widthBlockIdx = (blockIdx %
                     (numHeightBlock * numWidthBlock)) % numWidthBlock
    tid = tl.arange(0, headDim)
    VECSIZE: tl.constexpr = triton.next_power_of_2(widthBlockSize)
    local_width_indices = tl.arange(0, VECSIZE).to(tl.int64)
    for fb in tl.static_range(frameBlockSize):
        frameIdx = fb + frameBlockidx * frameBlockSize
        for hb in tl.static_range(heightBlockSize):
            heightIdx = hb + heightBlockIdx * heightBlockSize
            if frameIdx < frameDim and heightIdx < heightDim:
                widthIdx = widthBlockIdx * widthBlockSize + local_width_indices
                valid_width = local_width_indices < widthBlockSize
                widthIdxMask = valid_width & (widthIdx < widthDim)
                bblockIdx = fb * heightBlockSize * widthBlockSize + \
                            hb * widthBlockSize + \
                            local_width_indices[:, None]
                bblockIdx = tl.where(valid_width[:, None], bblockIdx, 0)
                oIdx = bszIdx * stride_o0 + blockIdx * stride_o1 + bblockIdx * stride_o2 + headListIdx * stride_o3 + tid[
                    None, :] * stride_o4
                new_oIdx = (
                    bszIdx * stride_new_o0 +
                    (frameIdx * heightDim * widthDim + heightIdx * widthDim +
                     widthIdx[:, None]) * stride_new_o1 +
                    headIdx * stride_new_o2 + tid[None, :] * stride_new_o3)
                oBlock = tl.load(o + oIdx, mask=widthIdxMask[:, None], other=0)
                tl.store(new_o + new_oIdx, oBlock, mask=widthIdxMask[:, None])


def ReorderBackTensorBSNH(o_new: torch.Tensor, o: torch.Tensor,
                          headIndices: torch.Tensor, seqlenDim3, blocksSize,
                          numBlocks):

    bsz, _, _, _, head_dim = o.size()
    _, _, num_heads, _ = o_new.size()
    numHeadList = headIndices.numel()
    frameDim, heightDim, widthDim = seqlenDim3
    frameBlockSize, heightBlockSize, widthBlockSize = blocksSize
    NumFrameBlock, NumHeightBlock, NumWidthBlock = numBlocks
    numBlock = NumFrameBlock * NumHeightBlock * NumWidthBlock
    grid = (bsz, numHeadList, numBlock)
    headIndices = headIndices.to(o.device)
    reorder_back_kernel[grid](o, o_new, headIndices, *o.stride(),
                              *o_new.stride(), frameDim, frameBlockSize,
                              NumFrameBlock, heightDim, heightBlockSize,
                              NumHeightBlock, widthDim, widthBlockSize,
                              NumWidthBlock, num_heads, head_dim)
    return o_new


@triton.jit
def reorder_head_first_kernel(
    k,  #[bsz,seqlen,numHead,headDim]
    new_k,  #[bsz,numHeadList,numBlock,blockSize,headDim]
    headIndices,  #[numListLength]
    padding_val: tl.constexpr,
    stride_k0: tl.constexpr,
    stride_k1: tl.constexpr,
    stride_k2: tl.constexpr,
    stride_k3: tl.constexpr,
    stride_new_k0,
    stride_new_k1,
    stride_new_k2,
    stride_new_k3,
    stride_new_k4,
    frameDim: tl.constexpr,
    heightDim: tl.constexpr,
    widthDim: tl.constexpr,
    frameBlockSize: tl.constexpr,
    heightBlockSize: tl.constexpr,
    widthBlockSize: tl.constexpr,
    numFrameBlock: tl.constexpr,
    numHeightBlock: tl.constexpr,
    numWidthBlock: tl.constexpr,
    numHeads: tl.constexpr,
    headDim: tl.constexpr,
):
    DTYPE: tl.constexpr = k.dtype.element_ty
    bszIdx = tl.program_id(0).to(tl.int64)
    headListIdx = tl.program_id(1).to(tl.int64)
    headIdx = tl.load(headIndices + headListIdx).to(tl.int64)
    blockIdx = tl.program_id(2).to(tl.int64)
    frameBlockidx = blockIdx // (numHeightBlock * numWidthBlock)
    heightBlockIdx = (blockIdx %
                      (numHeightBlock * numWidthBlock)) // numWidthBlock
    widthBlockIdx = (blockIdx %
                     (numHeightBlock * numWidthBlock)) % numWidthBlock
    tid = tl.arange(0, headDim)
    VECSIZE: tl.constexpr = triton.next_power_of_2(widthBlockSize)
    local_width_indices = tl.arange(0, VECSIZE).to(tl.int64)
    for fb in tl.static_range(frameBlockSize):
        frameIdx = fb + frameBlockidx * frameBlockSize
        for hb in tl.static_range(heightBlockSize):
            heightIdx = hb + heightBlockIdx * heightBlockSize
            widthIdx = widthBlockIdx * widthBlockSize + local_width_indices
            valid_width = local_width_indices < widthBlockSize
            widthIdxMask = valid_width & (widthIdx < widthDim)

            kIdx = (bszIdx * stride_k0 +
                    (frameIdx * heightDim * widthDim + heightIdx * widthDim +
                     widthIdx[:, None]) * stride_k1 + headIdx * stride_k2 +
                    tid[None, :] * stride_k3)
            if frameIdx < frameDim and heightIdx < heightDim:
                kBlock = tl.load(k + kIdx,
                                 mask=widthIdxMask[:, None],
                                 other=padding_val)
            else:
                kBlock = tl.full((VECSIZE, headDim), padding_val, dtype=DTYPE)

            bblockIdx = fb * heightBlockSize * widthBlockSize + \
                        hb * widthBlockSize + \
                        local_width_indices[:, None]
            bblockIdx = tl.where(valid_width[:, None], bblockIdx, 0)

            new_kIdx = (bszIdx * stride_new_k0 + headListIdx * stride_new_k1 +
                        blockIdx * stride_new_k2 + bblockIdx * stride_new_k3 +
                        tid[None, :] * stride_new_k4)

            tl.store(new_k + new_kIdx,
                     kBlock,
                     mask=(local_width_indices < widthBlockSize)[:, None])


def ReorderTensorBNSH(k_new: torch.Tensor, k: torch.Tensor,
                      headIndices: torch.Tensor, padding_val: float,
                      seqlenDim3, blocksSize, numBlocks):
    bsz, seqlen, numHeads, headDim = k.size()
    headIndices = headIndices.to(k.device).contiguous()
    numHeadList = headIndices.numel()
    assert seqlen == prod(seqlenDim3), f"seqlen {seqlen} != {prod(seqlenDim3)}"
    numBlock = prod(numBlocks)
    grid = (bsz, numHeadList, numBlock)

    reorder_head_first_kernel[grid](
        k,
        k_new,
        headIndices,
        padding_val,
        *k.stride(),
        *k_new.stride(),
        *seqlenDim3,
        *blocksSize,
        *numBlocks,
        numHeads,
        headDim,
    )
    return k_new


@triton.jit
def reorder_back_head_first_kernel(
    o,  # [bsz,numHeadList,numBlock,blockSize,headDim]
    new_o,  # [bsz,seqlen,num_heads,headDim]
    headIndices,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    stride_new_o0: tl.constexpr,
    stride_new_o1: tl.constexpr,
    stride_new_o2: tl.constexpr,
    stride_new_o3: tl.constexpr,
    frameDim: tl.constexpr,
    frameBlockSize: tl.constexpr,
    numFrameBlock: tl.constexpr,
    heightDim: tl.constexpr,
    heightBlockSize: tl.constexpr,
    numHeightBlock: tl.constexpr,
    widthDim: tl.constexpr,
    widthBlockSize: tl.constexpr,
    numWidthBlock: tl.constexpr,
    numHeads: tl.constexpr,
    headDim: tl.constexpr,
):
    bszIdx = tl.program_id(0).to(tl.int64)
    headListIdx = tl.program_id(1).to(tl.int64)
    headIdx = tl.load(headIndices + headListIdx).to(tl.int64)
    blockIdx = tl.program_id(2).to(tl.int64)
    frameBlockidx = blockIdx // (numHeightBlock * numWidthBlock)
    heightBlockIdx = (blockIdx %
                      (numHeightBlock * numWidthBlock)) // numWidthBlock
    widthBlockIdx = (blockIdx %
                     (numHeightBlock * numWidthBlock)) % numWidthBlock
    tid = tl.arange(0, headDim)
    VECSIZE: tl.constexpr = triton.next_power_of_2(widthBlockSize)
    local_width_indices = tl.arange(0, VECSIZE).to(tl.int64)
    for fb in tl.static_range(frameBlockSize):
        frameIdx = fb + frameBlockidx * frameBlockSize
        for hb in tl.static_range(heightBlockSize):
            heightIdx = hb + heightBlockIdx * heightBlockSize
            if frameIdx < frameDim and heightIdx < heightDim:
                widthIdx = widthBlockIdx * widthBlockSize + local_width_indices
                valid_width = local_width_indices < widthBlockSize
                widthIdxMask = valid_width & (widthIdx < widthDim)
                bblockIdx = fb * heightBlockSize * widthBlockSize + \
                            hb * widthBlockSize + \
                            local_width_indices[:, None]
                bblockIdx = tl.where(valid_width[:, None], bblockIdx, 0)
                oIdx = bszIdx * stride_o0 + headListIdx * stride_o1 + blockIdx * stride_o2 + bblockIdx * stride_o3 + tid[
                    None, :] * stride_o4
                new_oIdx = (
                    bszIdx * stride_new_o0 +
                    (frameIdx * heightDim * widthDim + heightIdx * widthDim +
                     widthIdx[:, None]) * stride_new_o1 +
                    headIdx * stride_new_o2 + tid[None, :] * stride_new_o3)
                oBlock = tl.load(o + oIdx, mask=widthIdxMask[:, None], other=0)
                tl.store(new_o + new_oIdx, oBlock, mask=widthIdxMask[:, None])


def ReorderBackTensorBNSH(o_new: torch.Tensor, o: torch.Tensor,
                          headIndices: torch.Tensor, seqlenDim3, blocksSize,
                          numBlocks):

    bsz, _, _, _, head_dim = o.size()
    _, _, num_heads, _ = o_new.size()
    numHeadList = headIndices.numel()
    frameDim, heightDim, widthDim = seqlenDim3
    frameBlockSize, heightBlockSize, widthBlockSize = blocksSize
    NumFrameBlock, NumHeightBlock, NumWidthBlock = numBlocks
    numBlock = NumFrameBlock * NumHeightBlock * NumWidthBlock
    grid = (bsz, numHeadList, numBlock)
    headIndices = headIndices.to(o.device)
    reorder_back_head_first_kernel[grid](
        o, o_new, headIndices, *o.stride(), *o_new.stride(), frameDim,
        frameBlockSize, NumFrameBlock, heightDim, heightBlockSize,
        NumHeightBlock, widthDim, widthBlockSize, NumWidthBlock, num_heads,
        head_dim)
    return o_new


@triton.jit
def reorder_text_kernel(
    k,  #[bsz,seqlen+textLength,numHead,headDim]
    new_k,  #[bsz,numBlock+textBlocks,blockSize,numHeadList,headDim]
    headIndices,  #[numListLength]
    realSeqlen: tl.constexpr,
    padding_val: tl.constexpr,
    stride_k0: tl.constexpr,
    stride_k1: tl.constexpr,
    stride_k2: tl.constexpr,
    stride_k3: tl.constexpr,
    stride_new_k0,
    stride_new_k1,
    stride_new_k2,
    stride_new_k3,
    stride_new_k4,
    frameDim: tl.constexpr,
    heightDim: tl.constexpr,
    widthDim: tl.constexpr,
    frameBlockSize: tl.constexpr,
    heightBlockSize: tl.constexpr,
    widthBlockSize: tl.constexpr,
    numFrameBlock: tl.constexpr,
    numHeightBlock: tl.constexpr,
    numWidthBlock: tl.constexpr,
    numHeads: tl.constexpr,
    headDim: tl.constexpr,
):
    DTYPE: tl.constexpr = k.dtype.element_ty
    bszIdx = tl.program_id(0).to(tl.int64)
    headListIdx = tl.program_id(1).to(tl.int64)
    headIdx = tl.load(headIndices + headListIdx).to(tl.int64)
    blockIdx = tl.program_id(2).to(tl.int64)
    vnumBlock: tl.constexpr = numFrameBlock * numHeightBlock * numWidthBlock
    blockSize: tl.constexpr = frameBlockSize * heightBlockSize * widthBlockSize
    vseqlen: tl.constexpr = frameDim * heightDim * widthDim
    tid = tl.arange(0, headDim)
    if blockIdx >= vnumBlock:
        tblockIdx = blockIdx - vnumBlock
        VECSIZE: tl.constexpr = triton.next_power_of_2(blockSize)
        local_indices = tl.arange(0, VECSIZE).to(tl.int64)
        seqlenIdx = local_indices + vseqlen + tblockIdx * blockSize
        kBlockIdx = (bszIdx * stride_k0 + seqlenIdx[:, None] * stride_k1 +
                     headIdx * stride_k2 + tid[None, :] * stride_k3)
        kBlock0 = tl.load(k + kBlockIdx,
                          mask=(seqlenIdx[:, None] < realSeqlen),
                          other=padding_val)
        new_kblockIdx = (bszIdx * stride_new_k0 +
                         blockIdx * stride_new_k1 +
                         local_indices[:, None] * stride_new_k2 +
                         headListIdx * stride_new_k3 +
                         tid[None, :] * stride_new_k4)
        tl.store(new_k + new_kblockIdx,
                 kBlock0,
                 mask=(local_indices < blockSize)[:, None])
    else:
        frameBlockidx = blockIdx // (numHeightBlock * numWidthBlock)
        heightBlockIdx = (blockIdx %
                          (numHeightBlock * numWidthBlock)) // numWidthBlock
        widthBlockIdx = (blockIdx %
                         (numHeightBlock * numWidthBlock)) % numWidthBlock

        VECSIZE: tl.constexpr = triton.next_power_of_2(widthBlockSize)
        local_width_indices = tl.arange(0, VECSIZE).to(tl.int64)
        for fb in tl.static_range(frameBlockSize):
            frameIdx = fb + frameBlockidx * frameBlockSize
            for hb in tl.static_range(heightBlockSize):
                heightIdx = hb + heightBlockIdx * heightBlockSize
                widthIdx = widthBlockIdx * widthBlockSize + local_width_indices
                valid_width = local_width_indices < widthBlockSize
                widthIdxMask = valid_width & (widthIdx < widthDim)

                kIdx = (
                    bszIdx * stride_k0 +
                    (frameIdx * heightDim * widthDim + heightIdx * widthDim +
                     widthIdx[:, None]) * stride_k1 + headIdx * stride_k2 +
                    tid[None, :] * stride_k3)
                if frameIdx < frameDim and heightIdx < heightDim:
                    kBlock = tl.load(k + kIdx,
                                     mask=widthIdxMask[:, None],
                                     other=padding_val)
                else:
                    kBlock = tl.full((VECSIZE, headDim),
                                     padding_val,
                                     dtype=DTYPE)

                bblockIdx = fb * heightBlockSize * widthBlockSize + \
                            hb * widthBlockSize + \
                            local_width_indices[:, None]
                bblockIdx = tl.where(valid_width[:, None], bblockIdx, 0)

                new_kIdx = (bszIdx * stride_new_k0 +
                            blockIdx * stride_new_k1 +
                            bblockIdx * stride_new_k2 +
                            headListIdx * stride_new_k3 +
                            tid[None, :] * stride_new_k4)

                tl.store(new_k + new_kIdx,
                         kBlock,
                         mask=(local_width_indices < widthBlockSize)[:, None])


@triton.jit
def reorder_head_first_text_kernel(
    k,  #[bsz,seqlen+textLength,numHead,headDim]
    new_k,  #[bsz,numHeadList,numBlock+textBlocks,blockSize,headDim]
    headIndices,  #[numListLength]
    realSeqlen: tl.constexpr,
    padding_val: tl.constexpr,
    stride_k0: tl.constexpr,
    stride_k1: tl.constexpr,
    stride_k2: tl.constexpr,
    stride_k3: tl.constexpr,
    stride_new_k0,
    stride_new_k1,
    stride_new_k2,
    stride_new_k3,
    stride_new_k4,
    frameDim: tl.constexpr,
    heightDim: tl.constexpr,
    widthDim: tl.constexpr,
    frameBlockSize: tl.constexpr,
    heightBlockSize: tl.constexpr,
    widthBlockSize: tl.constexpr,
    numFrameBlock: tl.constexpr,
    numHeightBlock: tl.constexpr,
    numWidthBlock: tl.constexpr,
    numHeads: tl.constexpr,
    headDim: tl.constexpr,
):
    DTYPE: tl.constexpr = k.dtype.element_ty
    bszIdx = tl.program_id(0).to(tl.int64)
    headListIdx = tl.program_id(1).to(tl.int64)
    headIdx = tl.load(headIndices + headListIdx).to(tl.int64)
    blockIdx = tl.program_id(2).to(tl.int64)
    vnumBlock: tl.constexpr = numFrameBlock * numHeightBlock * numWidthBlock
    blockSize: tl.constexpr = frameBlockSize * heightBlockSize * widthBlockSize
    vseqlen: tl.constexpr = frameDim * heightDim * widthDim
    tid = tl.arange(0, headDim)
    if blockIdx >= vnumBlock:
        tblockIdx = blockIdx - vnumBlock
        VECSIZE: tl.constexpr = triton.next_power_of_2(blockSize)
        local_indices = tl.arange(0, VECSIZE).to(tl.int64)
        seqlenIdx = local_indices + vseqlen + tblockIdx * blockSize
        kBlockIdx = (bszIdx * stride_k0 + seqlenIdx[:, None] * stride_k1 +
                     headIdx * stride_k2 + tid[None, :] * stride_k3)
        kBlock0 = tl.load(k + kBlockIdx,
                          mask=(seqlenIdx[:, None] < realSeqlen),
                          other=padding_val)
        new_kblockIdx = (bszIdx * stride_new_k0 + headListIdx * stride_new_k1 +
                         blockIdx * stride_new_k2 +
                         local_indices[:, None] * stride_new_k3 +
                         tid[None, :] * stride_new_k4)
        tl.store(new_k + new_kblockIdx,
                 kBlock0,
                 mask=(local_indices < blockSize)[:, None])
    else:
        frameBlockidx = blockIdx // (numHeightBlock * numWidthBlock)
        heightBlockIdx = (blockIdx %
                          (numHeightBlock * numWidthBlock)) // numWidthBlock
        widthBlockIdx = (blockIdx %
                         (numHeightBlock * numWidthBlock)) % numWidthBlock

        VECSIZE: tl.constexpr = triton.next_power_of_2(widthBlockSize)
        local_width_indices = tl.arange(0, VECSIZE).to(tl.int64)
        for fb in tl.static_range(frameBlockSize):
            frameIdx = fb + frameBlockidx * frameBlockSize
            for hb in tl.static_range(heightBlockSize):
                heightIdx = hb + heightBlockIdx * heightBlockSize
                widthIdx = widthBlockIdx * widthBlockSize + local_width_indices
                valid_width = local_width_indices < widthBlockSize
                widthIdxMask = valid_width & (widthIdx < widthDim)

                kIdx = (
                    bszIdx * stride_k0 +
                    (frameIdx * heightDim * widthDim + heightIdx * widthDim +
                     widthIdx[:, None]) * stride_k1 + headIdx * stride_k2 +
                    tid[None, :] * stride_k3)
                if frameIdx < frameDim and heightIdx < heightDim:
                    kBlock = tl.load(k + kIdx,
                                     mask=widthIdxMask[:, None],
                                     other=padding_val)
                else:
                    kBlock = tl.full((VECSIZE, headDim),
                                     padding_val,
                                     dtype=DTYPE)

                bblockIdx = fb * heightBlockSize * widthBlockSize + \
                            hb * widthBlockSize + \
                            local_width_indices[:, None]
                bblockIdx = tl.where(valid_width[:, None], bblockIdx, 0)

                new_kIdx = (bszIdx * stride_new_k0 +
                            headListIdx * stride_new_k1 +
                            blockIdx * stride_new_k2 +
                            bblockIdx * stride_new_k3 +
                            tid[None, :] * stride_new_k4)

                tl.store(new_k + new_kIdx,
                         kBlock,
                         mask=(local_width_indices < widthBlockSize)[:, None])



@triton.jit
def reorder_back_text_kernel(
    o,  # [bsz,numBlock+textBlocks,blockSize,numHeadList,headDim]
    new_o,  # [bsz,seqlen,num_heads,headDim]
    headIndices,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    stride_new_o0: tl.constexpr,
    stride_new_o1: tl.constexpr,
    stride_new_o2: tl.constexpr,
    stride_new_o3: tl.constexpr,
    frameDim: tl.constexpr,
    frameBlockSize: tl.constexpr,
    numFrameBlock: tl.constexpr,
    heightDim: tl.constexpr,
    heightBlockSize: tl.constexpr,
    numHeightBlock: tl.constexpr,
    widthDim: tl.constexpr,
    widthBlockSize: tl.constexpr,
    numWidthBlock: tl.constexpr,
    numHeads: tl.constexpr,
    headDim: tl.constexpr,
):
    bszIdx = tl.program_id(0).to(tl.int64)
    headListIdx = tl.program_id(1).to(tl.int64)
    headIdx = tl.load(headIndices + headListIdx).to(tl.int64)
    blockIdx = tl.program_id(2).to(tl.int64)
    vnumBlock: tl.constexpr = numFrameBlock * numHeightBlock * numWidthBlock
    if blockIdx >= vnumBlock:
        vseqlen: tl.constexpr = frameDim * heightDim * widthDim
        blockSize: tl.constexpr = frameBlockSize * heightBlockSize * widthBlockSize
        VECSIZE: tl.constexpr = triton.next_power_of_2(blockSize)
        local_indices = tl.arange(0, VECSIZE).to(tl.int64)
        tid = tl.arange(0, headDim).to(tl.int64)
        seqlenIdx = local_indices + vseqlen + (blockIdx -
                                               vnumBlock) * blockSize
        oIdx = bszIdx * stride_o0 + blockIdx * stride_o1 + local_indices[:, None] * stride_o2 + headListIdx * stride_o3 + tid[
            None, :] * stride_o4
        new_oIdx = (bszIdx * stride_new_o0 +
                    seqlenIdx[:, None] * stride_new_o1 +
                    headIdx * stride_new_o2 + tid[None, :] * stride_new_o3)
        oBlock = tl.load(o + oIdx)
        tl.store(new_o + new_oIdx, oBlock)
    else:
        frameBlockidx = blockIdx // (numHeightBlock * numWidthBlock)
        heightBlockIdx = (blockIdx %
                          (numHeightBlock * numWidthBlock)) // numWidthBlock
        widthBlockIdx = (blockIdx %
                         (numHeightBlock * numWidthBlock)) % numWidthBlock
        tid = tl.arange(0, headDim).to(tl.int64)
        VECSIZE: tl.constexpr = triton.next_power_of_2(widthBlockSize)
        local_width_indices = tl.arange(0, VECSIZE).to(tl.int64)
        for fb in tl.static_range(frameBlockSize):
            frameIdx = fb + frameBlockidx * frameBlockSize
            for hb in tl.static_range(heightBlockSize):
                heightIdx = hb + heightBlockIdx * heightBlockSize
                if frameIdx < frameDim and heightIdx < heightDim:
                    widthIdx = widthBlockIdx * widthBlockSize + local_width_indices
                    valid_width = local_width_indices < widthBlockSize
                    widthIdxMask = valid_width & (widthIdx < widthDim)
                    bblockIdx = fb * heightBlockSize * widthBlockSize + \
                                hb * widthBlockSize + \
                                local_width_indices[:, None]
                    bblockIdx = tl.where(valid_width[:, None], bblockIdx, 0)
                    oIdx = bszIdx * stride_o0 + blockIdx * stride_o1 + bblockIdx * stride_o2 + headListIdx * stride_o3 + tid[
                        None, :] * stride_o4
                    new_oIdx = (bszIdx * stride_new_o0 +
                                (frameIdx * heightDim * widthDim +
                                 heightIdx * widthDim + widthIdx[:, None]) *
                                stride_new_o1 + headIdx * stride_new_o2 +
                                tid[None, :] * stride_new_o3)
                    oBlock = tl.load(o + oIdx,
                                     mask=widthIdxMask[:, None],
                                     other=0)
                    tl.store(new_o + new_oIdx,
                             oBlock,
                             mask=widthIdxMask[:, None])


@triton.jit
def reorder_back_head_first_text_kernel(
    o,  # [bsz,numHeadList,numBlock+textBlocks,blockSize,headDim]
    new_o,  # [bsz,seqlen,num_heads,headDim]
    headIndices,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    stride_new_o0: tl.constexpr,
    stride_new_o1: tl.constexpr,
    stride_new_o2: tl.constexpr,
    stride_new_o3: tl.constexpr,
    frameDim: tl.constexpr,
    frameBlockSize: tl.constexpr,
    numFrameBlock: tl.constexpr,
    heightDim: tl.constexpr,
    heightBlockSize: tl.constexpr,
    numHeightBlock: tl.constexpr,
    widthDim: tl.constexpr,
    widthBlockSize: tl.constexpr,
    numWidthBlock: tl.constexpr,
    numHeads: tl.constexpr,
    headDim: tl.constexpr,
):
    bszIdx = tl.program_id(0).to(tl.int64)
    headListIdx = tl.program_id(1).to(tl.int64)
    headIdx = tl.load(headIndices + headListIdx).to(tl.int64)
    blockIdx = tl.program_id(2).to(tl.int64)
    vnumBlock: tl.constexpr = numFrameBlock * numHeightBlock * numWidthBlock
    if blockIdx >= vnumBlock:
        vseqlen: tl.constexpr = frameDim * heightDim * widthDim
        blockSize: tl.constexpr = frameBlockSize * heightBlockSize * widthBlockSize
        VECSIZE: tl.constexpr = triton.next_power_of_2(blockSize)
        local_indices = tl.arange(0, VECSIZE).to(tl.int64)
        tid = tl.arange(0, headDim).to(tl.int64)
        seqlenIdx = local_indices + vseqlen + (blockIdx -
                                               vnumBlock) * blockSize
        oIdx = bszIdx * stride_o0 + headListIdx * stride_o1 + blockIdx * stride_o2 + local_indices[:, None] * stride_o3 + tid[
            None, :] * stride_o4
        new_oIdx = (bszIdx * stride_new_o0 +
                    seqlenIdx[:, None] * stride_new_o1 +
                    headIdx * stride_new_o2 + tid[None, :] * stride_new_o3)
        oBlock = tl.load(o + oIdx)
        tl.store(new_o + new_oIdx, oBlock)
    else:
        frameBlockidx = blockIdx // (numHeightBlock * numWidthBlock)
        heightBlockIdx = (blockIdx %
                          (numHeightBlock * numWidthBlock)) // numWidthBlock
        widthBlockIdx = (blockIdx %
                         (numHeightBlock * numWidthBlock)) % numWidthBlock
        tid = tl.arange(0, headDim).to(tl.int64)
        VECSIZE: tl.constexpr = triton.next_power_of_2(widthBlockSize)
        local_width_indices = tl.arange(0, VECSIZE).to(tl.int64)
        for fb in tl.static_range(frameBlockSize):
            frameIdx = fb + frameBlockidx * frameBlockSize
            for hb in tl.static_range(heightBlockSize):
                heightIdx = hb + heightBlockIdx * heightBlockSize
                if frameIdx < frameDim and heightIdx < heightDim:
                    widthIdx = widthBlockIdx * widthBlockSize + local_width_indices
                    valid_width = local_width_indices < widthBlockSize
                    widthIdxMask = valid_width & (widthIdx < widthDim)
                    bblockIdx = fb * heightBlockSize * widthBlockSize + \
                                hb * widthBlockSize + \
                                local_width_indices[:, None]
                    bblockIdx = tl.where(valid_width[:, None], bblockIdx, 0)
                    oIdx = bszIdx * stride_o0 + headListIdx * stride_o1 + blockIdx * stride_o2 + bblockIdx * stride_o3 + tid[
                        None, :] * stride_o4
                    new_oIdx = (bszIdx * stride_new_o0 +
                                (frameIdx * heightDim * widthDim +
                                 heightIdx * widthDim + widthIdx[:, None]) *
                                stride_new_o1 + headIdx * stride_new_o2 +
                                tid[None, :] * stride_new_o3)
                    oBlock = tl.load(o + oIdx,
                                     mask=widthIdxMask[:, None],
                                     other=0)
                    tl.store(new_o + new_oIdx,
                             oBlock,
                             mask=widthIdxMask[:, None])


def ReorderTensorTextBSNH(
    k_new: torch.Tensor,
    k: torch.Tensor,
    headIndices: torch.Tensor,
    padding_val: float,
    seqlenDim3: tuple,
    blocksSize: tuple,
    numBlocks: tuple,
    realSeqlen: int,
):
    bsz, seqlen, numHeads, headDim = k.size()
    headIndices = headIndices.to(k.device).contiguous()
    numHeadList = headIndices.numel()
    blockSize = prod(blocksSize)
    cxtSeqlen = seqlen - prod(seqlenDim3)
    assert cxtSeqlen % blockSize == 0, f"cxtSeqlen {cxtSeqlen} must be divisible by blockSize {blockSize}"
    numBlock = prod(numBlocks) + cxtSeqlen // blockSize
    grid = (bsz, numHeadList, numBlock)

    reorder_text_kernel[grid](
        k,
        k_new,
        headIndices,
        realSeqlen,
        padding_val,
        *k.stride(),
        *k_new.stride(),
        *seqlenDim3,
        *blocksSize,
        *numBlocks,
        numHeads,
        headDim,
    )
    return k_new


def ReorderTensorTextBNSH(
    k_new: torch.Tensor,
    k: torch.Tensor,
    headIndices: torch.Tensor,
    padding_val: float,
    seqlenDim3: tuple,
    blocksSize: tuple,
    numBlocks: tuple,
    realSeqlen: int,
):
    bsz, seqlen, numHeads, headDim = k.size()
    headIndices = headIndices.to(k.device).contiguous()
    numHeadList = headIndices.numel()
    blockSize = prod(blocksSize)
    cxtSeqlen = seqlen - prod(seqlenDim3)
    assert cxtSeqlen % blockSize == 0, f"cxtSeqlen {cxtSeqlen} must be divisible by blockSize {blockSize}"
    numBlock = prod(numBlocks) + cxtSeqlen // blockSize
    grid = (bsz, numHeadList, numBlock)

    reorder_head_first_text_kernel[grid](
        k,
        k_new,
        headIndices,
        realSeqlen,
        padding_val,
        *k.stride(),
        *k_new.stride(),
        *seqlenDim3,
        *blocksSize,
        *numBlocks,
        numHeads,
        headDim,
    )
    return k_new


def ReorderBackTensorTextBSNH(
    o_new: torch.Tensor,
    o: torch.Tensor,
    headIndices: torch.Tensor,
    seqlenDim3: tuple,
    blocksSize: tuple,
    numBlocks: tuple,
):

    bsz, _, _, _, head_dim = o.size()
    _, seqlen, num_heads, _ = o_new.size()
    numHeadList = headIndices.numel()
    cxtSeqlen = seqlen - prod(seqlenDim3)
    assert cxtSeqlen % prod(
        blocksSize) == 0, "cxtSeqlen must be divisible by prod(blocksSize)"
    frameDim, heightDim, widthDim = seqlenDim3
    frameBlockSize, heightBlockSize, widthBlockSize = blocksSize
    NumFrameBlock, NumHeightBlock, NumWidthBlock = numBlocks
    numBlock = NumFrameBlock * NumHeightBlock * NumWidthBlock + cxtSeqlen // prod(
        blocksSize)
    grid = (bsz, numHeadList, numBlock)
    headIndices = headIndices.to(o.device)
    reorder_back_text_kernel[grid](
        o, o_new, headIndices, *o.stride(), *o_new.stride(), frameDim,
        frameBlockSize, NumFrameBlock, heightDim, heightBlockSize,
        NumHeightBlock, widthDim, widthBlockSize, NumWidthBlock, num_heads,
        head_dim)
    return o_new


def ReorderBackTensorTextBNSH(
    o_new: torch.Tensor,
    o: torch.Tensor,
    headIndices: torch.Tensor,
    seqlenDim3: tuple,
    blocksSize: tuple,
    numBlocks: tuple,
):

    bsz, _, _, _, head_dim = o.size()
    _, seqlen, num_heads, _ = o_new.size()
    numHeadList = headIndices.numel()
    cxtSeqlen = seqlen - prod(seqlenDim3)
    assert cxtSeqlen % prod(
        blocksSize) == 0, "cxtSeqlen must be divisible by prod(blocksSize)"
    frameDim, heightDim, widthDim = seqlenDim3
    frameBlockSize, heightBlockSize, widthBlockSize = blocksSize
    NumFrameBlock, NumHeightBlock, NumWidthBlock = numBlocks
    numBlock = NumFrameBlock * NumHeightBlock * NumWidthBlock + cxtSeqlen // prod(
        blocksSize)
    grid = (bsz, numHeadList, numBlock)
    headIndices = headIndices.to(o.device)
    reorder_back_head_first_text_kernel[grid](
        o, o_new, headIndices, *o.stride(), *o_new.stride(), frameDim,
        frameBlockSize, NumFrameBlock, heightDim, heightBlockSize,
        NumHeightBlock, widthDim, widthBlockSize, NumWidthBlock, num_heads,
        head_dim)
    return o_new
