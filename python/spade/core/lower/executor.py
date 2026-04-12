from typing import Callable, Union, Optional, List
import torch
from torch.nn.functional import softmax
from math import prod

from spade_utils import scatter_mask, static_sink_diag_set

from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.utils.common import StaticMaskSet
from spade.core.lower.utils.reorder import (
    ReorderBackTensorBSNH,
    ReorderBackTensorBNSH,
    ReorderBackTensorTextBSNH,
    ReorderBackTensorTextBNSH,
)
import importlib


class SummarizerExecutor:

    def __init__(self, sparse_config: SparseHeadConfig,
                 layout: str = "bsnh") -> None:
        self.sparse_config = sparse_config
        self.layout = layout

    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 head_ids: torch.Tensor,
                 block_q: torch.Tensor,
                 block_k: torch.Tensor,
                 block_v: torch.Tensor,
                 realSeqlen: Optional[int] = None):
        raise NotImplementedError('SummarizerExecutor is not implemented')


class EstimatorExecutor:

    def __init__(self, sparse_config: SparseHeadConfig,
                 layout: str = "bsnh") -> None:
        self.sparse_config = sparse_config
        self.layout = layout
        self.dync_enable = sparse_config.dync_enable
        if self.dync_enable:
            self.symbol_inter_estimator = sparse_config.symbol_inter_estimator
        self.neg_inf = torch.tensor(-1e10, dtype=sparse_config.attn_dtype)

    def _check(self, tq: torch.Tensor):
        if prod(self.sparse_config.seqlenTuple) != tq.shape[2]:
            return False
        return True

    @staticmethod
    def inter_topk(estimates: torch.Tensor, mask: torch.Tensor,
                   topk_size: int):
        topk_size = min(topk_size, estimates.shape[-1])
        index = torch.topk(estimates, topk_size, dim=-1).indices
        mask.scatter_(-1, index, True)

    @staticmethod
    def inter_topk_tensor(estimates: torch.Tensor, mask: torch.Tensor,
                          topk_size: torch.Tensor):
        topk_size_val = min(topk_size.max().item(), estimates.shape[-1])
        index = torch.topk(estimates, topk_size_val,
                           dim=-1).indices.to(torch.int32)
        scatter_mask(mask, index, topk_size)

    @staticmethod
    def inter_topp(estimates: torch.Tensor, mask: torch.Tensor, top_p: float):
        estimates = softmax(estimates, dim=-1)
        sorted_estimates, sorted_indices = torch.sort(estimates,
                                                      dim=-1,
                                                      descending=True)
        sorted_cumsum = torch.cumsum(
            sorted_estimates,
            dim=-1,
        )
        shifted_cumsum = sorted_cumsum - sorted_estimates
        new_mask = torch.zeros_like(mask)
        new_mask.scatter_(-1, sorted_indices, shifted_cumsum < top_p)
        mask.logical_or_(new_mask)

    @staticmethod
    def intra_topk(estimates: torch.Tensor, mask_e: torch.Tensor,
                   topk_size: int):
        topk_size = min(topk_size, estimates.shape[-1])
        index = torch.topk(estimates, topk_size, dim=-1).indices
        mask_e.scatter_(-1, index, True)

    def __call__(
        self,
        mask: torch.Tensor,
        inter_top_val: Optional[Union[Union[float, int], torch.Tensor]] = None,
        intra_top_val: Optional[Union[float, int]] = None,
        diag_width: Optional[Union[int, torch.Tensor]] = None,
        block_inter_q: Optional[dict[str, torch.Tensor]] = None,
        block_inter_k: Optional[dict[str, torch.Tensor]] = None,
        block_intra_q: Optional[torch.Tensor] = None,
        block_intra_k: Optional[torch.Tensor] = None,
        realSeqlen: Optional[int] = None,
    ) -> torch.Tensor:
        '''
        mask: (bsz, num_heads, qNumBlocks, kNumBlocks)
        inter_top_val: Union[float, int]
        intra_top_val: Union[float, int]
        diag_width: Union[int, torch.Tensor]
        block_inter_q: dict(str, (bsz, num_heads, qNumBlocks, hidden_dim))
        block_inter_k: dict(str, (bsz, num_heads, kNumBlocks, hidden_dim))
        block_intra_q: (bsz, num_heads, qNumBlocks)
        block_intra_k: (bsz, num_heads, kNumBlocks)
        realSeqlen: Optional[int] = None, for hunyuan model arch, the total length includes video length and text length
        '''

        if self.dync_enable:
            if self.sparse_config.inter_summarize_enable:
                inter_estimate = self.sparse_config.symbol_inter_estimator(
                    block_inter_q, block_inter_k)

                if self.sparse_config.inter_is_topk_mode:
                    if isinstance(inter_top_val, torch.Tensor):
                        EstimatorExecutor.inter_topk_tensor(
                            inter_estimate, mask, inter_top_val)
                    elif isinstance(inter_top_val, int):
                        EstimatorExecutor.inter_topk(inter_estimate, mask,
                                                     inter_top_val)
                    else:
                        raise TypeError(
                            'inter_top_val must be int or tensor(int) in topk mode'
                        )
                else:
                    assert isinstance(
                        inter_top_val,
                        float), 'inter_top_val must be float in topp mode'
                    assert inter_top_val > 0 and inter_top_val < 1, 'inter_top_val must be greater than 0 and less than 1'
                    EstimatorExecutor.inter_topp(inter_estimate, mask,
                                                 inter_top_val)

            if self.sparse_config.intra_summarize_enable and intra_top_val < 0.99:
                if self.sparse_config.intra_is_topk_mode:
                    assert isinstance(
                        intra_top_val,
                        int), 'intra_top_val must be int in topk mode'
                    intra_qk = torch.zeros_like(block_intra_q,
                                                dtype=torch.bool)
                    EstimatorExecutor.intra_topk(block_intra_q + block_intra_k,
                                                 intra_qk, intra_top_val)

                else:
                    assert isinstance(
                        intra_top_val,
                        float), 'intra_top_val must be float in topp mode'
                    assert intra_top_val > 0 and intra_top_val < 1, 'intra_top_val must be greater than 0 and less than 1'

                    intra_qk = (block_intra_q > intra_top_val)

                mask.logical_and_((~intra_qk).unsqueeze_(-1))

            if isinstance(diag_width, int):
                StaticMaskSet(mask, 1, self.sparse_config.fixed_sink_width,
                              diag_width)
            elif isinstance(diag_width, torch.Tensor):
                static_sink_diag_set(mask, diag_width,
                                     self.sparse_config.fixed_sink_width)
            else:
                raise TypeError('diag_width must be int or tensor(int), got ', type(diag_width))

            if self.sparse_config.context_length:
                propmtSeqlen = realSeqlen - self.sparse_config.seqlen
                qBlockSize = self.sparse_config.block_size_q_int
                kvBlockSize = self.sparse_config.block_size_kv_int
                propmtkvNumBlocks = (propmtSeqlen + kvBlockSize -
                                     1) // kvBlockSize
                propmtqNumBlocks = (propmtSeqlen + qBlockSize -
                                    1) // qBlockSize
                qNumBlocks = self.sparse_config.num_blocks_q_int
                kNumBlocks = self.sparse_config.num_blocks_kv_int
                mask[:, :, qNumBlocks:qNumBlocks + propmtqNumBlocks] = True
                mask[:, :, :, kNumBlocks:kNumBlocks + propmtkvNumBlocks] = True
                mask[:, :, :, kNumBlocks + propmtkvNumBlocks:] = False
                #TODO: BSA fix First block -1
                mask[:, :, qNumBlocks + propmtqNumBlocks:] = False
                mask[:, :, qNumBlocks + propmtqNumBlocks:, 0] = True


class AttentionExecutor:

    def __init__(self, sparse_config: SparseHeadConfig,
                 layout: str = "bsnh", backend: str = None) -> None:
        self.sparse_config = sparse_config
        self.layout = layout
        self.backend = backend
        self._blocksparse_flashattn, self.attention_layout = self._resolve_backend(
            backend, requested_layout=layout)

        if self.layout != self.attention_layout:
            raise ValueError(
                f"Backend '{self.backend}' requires attention_layout='{self.attention_layout}', got layout='{self.layout}'"
            )

    @staticmethod
    def _resolve_backend(backend: str, requested_layout: str = None):
        def _load_backend(name: str):
            module = importlib.import_module(
                f"spade.core.lower.block_sparse_attn.block_sparse_attn_{name}"
            )
            return module.blocksparse_flashattn, module.attention_layout

        def _try_backend(name: str):
            try:
                fn, layout = _load_backend(name)
                return fn, layout, None
            except Exception as exc:  # ImportError or build/runtime failures
                return None, None, exc

        if backend == "cuda":
            backend = None

        if backend is None:
            order = ["cutlass", "flex", "tk"]
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability in {(8, 0), (8, 9)}:
                    order = ["bnsh", "cutlass", "flex", "tk"]
                elif capability == (9, 0):
                    order = ["bnsh", "cutlass", "flex", "tk"]

            errors = {}
            for name in order:
                fn, layout, err = _try_backend(name)
                if fn is not None:
                    if requested_layout is not None and layout != requested_layout:
                        errors[name] = (
                            f"backend layout '{layout}' does not match requested "
                            f"layout '{requested_layout}'"
                        )
                        continue
                    return fn, layout
                errors[name] = err
            raise RuntimeError(
                f"No block sparse attention backend available; errors: {errors}"
            )

        if backend not in ["bnsh", "tk", "cutlass", "flex"]:
            raise ValueError(
                "Unsupported backend "
                f"'{backend}', expected one of ['bnsh', 'tk', 'cutlass', 'flex']"
            )
        fn, layout, err = _try_backend(backend)
        if fn is None:
            raise RuntimeError(
                f"Failed to load backend '{backend}': {err}"
            ) from err
        return fn, layout

    def _run_blocksparse(self, block_q: torch.Tensor, block_k: torch.Tensor,
                         block_v: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        return self._blocksparse_flashattn(block_q, block_k, block_v, mask)

    def __call__(self, res_o: torch.Tensor, head_ids: List[int],
                 block_q: torch.Tensor, block_k: torch.Tensor,
                 block_v: torch.Tensor, mask: torch.Tensor):
        '''
        res_o: (bsz,  seqlen, num_heads, hidden_dim)
        head_ids: (num_head_ids)
        block_q: (bsz, num_head_ids, qNumBlocks, qBlockSize,  hidden_dim) if layout == "bnsh" else (bsz, qNumBlocks, qBlockSize, num_head_ids, hidden_dim)
        block_k: (bsz, num_head_ids, kNumBlocks, kBlockSize, hidden_dim) if layout == "bnsh" else (bsz, kNumBlocks, kBlockSize, num_head_ids, hidden_dim)
        block_v: (bsz, num_head_ids, kNumBlocks, kBlockSize, hidden_dim) if layout == "bnsh" else (bsz, kNumBlocks, kBlockSize, num_head_ids, hidden_dim)
        mask: (bsz, num_head_ids, qNumBlocks, kNumBlocks)
        '''
        block_o = self._run_blocksparse(block_q, block_k, block_v, mask)
        if self.attention_layout == "bnsh":
            if self.sparse_config.context_length:
                ReorderBackTensorTextBNSH(res_o, block_o, head_ids,
                                          self.sparse_config.seqlen3d,
                                          self.sparse_config.block_size_q,
                                          self.sparse_config.num_blocks_q)
            else:
                ReorderBackTensorBNSH(res_o, block_o, head_ids,
                                      self.sparse_config.seqlen3d,
                                      self.sparse_config.block_size_q,
                                      self.sparse_config.num_blocks_q)
        else:
            if self.sparse_config.context_length:
                ReorderBackTensorTextBSNH(res_o, block_o, head_ids,
                                          self.sparse_config.seqlen3d,
                                          self.sparse_config.block_size_q,
                                          self.sparse_config.num_blocks_q)
            else:
                ReorderBackTensorBSNH(res_o, block_o, head_ids,
                                      self.sparse_config.seqlen3d,
                                      self.sparse_config.block_size_q,
                                      self.sparse_config.num_blocks_q)


class E2EExecutor:

    def __init__(self, sparse_config: SparseHeadConfig,
                 summarizer_exe: SummarizerExecutor,
                 estimator_exe: EstimatorExecutor,
                 attention_exe: AttentionExecutor) -> None:
        self.sparse_config = sparse_config
        self.summarizer_exe = summarizer_exe
        self.estimator_exe = estimator_exe
        self.attention_exe = attention_exe
        if hasattr(self.summarizer_exe,
                   "layout") and hasattr(self.attention_exe, "layout"):
            if self.summarizer_exe.layout != self.attention_exe.layout:
                raise ValueError(
                    f"Summarizer layout '{self.summarizer_exe.layout}' does not match attention layout '{self.attention_exe.layout}'"
                )

    def _gen_block_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       head_ids: List[int]):

        device = q.device
        dtype = q.dtype
        head_ids_torch = torch.tensor(head_ids,
                                      dtype=torch.int64,
                                      device=device)
        bsz, seqlen, num_heads, hidden_dim = q.shape
        num_head_lists = head_ids_torch.numel()

        kvNumBlocks = self.sparse_config.num_blocks_kv_int
        kvBlockSize = self.sparse_config.block_size_kv_int

        qNumBlocks = self.sparse_config.num_blocks_q_int
        qBlockSize = self.sparse_config.block_size_q_int

        if self.sparse_config.context_length:
            qNumBlocks += self.sparse_config.context_q_num_block
            kvNumBlocks += self.sparse_config.context_kv_num_block

        layout = getattr(self.attention_exe, "layout", "bnsh")
        if layout == "bsnh":
            block_q = torch.empty(bsz,
                                  qNumBlocks,
                                  qBlockSize,
                                  num_head_lists,
                                  hidden_dim,
                                  device=device,
                                  dtype=dtype)
            block_k = torch.empty(bsz,
                                  kvNumBlocks,
                                  kvBlockSize,
                                  num_head_lists,
                                  hidden_dim,
                                  device=device,
                                  dtype=dtype)
            block_v = torch.empty(bsz,
                                  kvNumBlocks,
                                  kvBlockSize,
                                  num_head_lists,
                                  hidden_dim,
                                  device=device,
                                  dtype=dtype)
        else:
            block_q = torch.empty(bsz,
                                  num_head_lists,
                                  qNumBlocks,
                                  qBlockSize,
                                  hidden_dim,
                                  device=device,
                                  dtype=dtype)
            block_k = torch.empty(bsz,
                                  num_head_lists,
                                  kvNumBlocks,
                                  kvBlockSize,
                                  hidden_dim,
                                  device=device,
                                  dtype=dtype)
            block_v = torch.empty(bsz,
                                  num_head_lists,
                                  kvNumBlocks,
                                  kvBlockSize,
                                  hidden_dim,
                                  device=device,
                                  dtype=dtype)

        dense_mask = torch.zeros(bsz,
                                 num_head_lists,
                                 qNumBlocks,
                                 kvNumBlocks,
                                 device=device,
                                 dtype=torch.bool)

        return block_q, block_k, block_v, dense_mask, head_ids_torch

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 res_o: torch.Tensor, head_ids: List[int],
                 inter_top_val: Union[float,
                                      torch.Tensor], intra_top_val: float,
                 diag_width: Union[int,
                                   torch.tensor], realSeqlen: Optional[int]):

        assert not (
            (realSeqlen is None) ^ (self.sparse_config.context_length is None)
        ), "realSeqlen and context_length must be both None or not None"
        if isinstance(diag_width, torch.Tensor):
            diag_width = diag_width.to(torch.int32)
        if isinstance(inter_top_val,
                      torch.Tensor) and self.sparse_config.inter_is_topk_mode:
            inter_top_val = inter_top_val.to(torch.int32)
        elif inter_top_val > 1:
            inter_top_val = int(inter_top_val)

        if intra_top_val > 1:
            intra_top_val = int(intra_top_val)
        block_q, block_k, block_v, dense_mask, head_ids = self._gen_block_qkv(
            q, k, v, head_ids)

        summarize_res_dict = self.summarizer_exe(q, k, v, head_ids, block_q,
                                                 block_k, block_v, realSeqlen)

        self.estimator_exe(**summarize_res_dict,
                           mask=dense_mask,
                           inter_top_val=inter_top_val,
                           intra_top_val=intra_top_val,
                           diag_width=diag_width,
                           realSeqlen=realSeqlen)
        self.attention_exe(res_o, head_ids, block_q, block_k, block_v,
                           dense_mask)
        sparse_rate = 1 - dense_mask.sum() / (dense_mask.numel())
        return sparse_rate

    def time_item(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        res_o: torch.Tensor,
        head_ids: List[int],
        inter_top_val: Union[float, int],
        intra_top_val: Union[float, int],
        diag_width: Union[int, torch.Tensor],
        realSeqlen: Optional[int],
        warmup: int = 5,
        repeat: int = 10,
    ):
        import time

        block_q, block_k, block_v, dense_mask, head_ids = self._gen_block_qkv(
            q, k, v, head_ids)

        # Warmup
        for _ in range(warmup):
            summarize_res_dict = self.summarizer_exe(q, k, v, head_ids,
                                                     block_q, block_k, block_v,
                                                     realSeqlen)
            self.estimator_exe(**summarize_res_dict,
                               mask=dense_mask,
                               inter_top_val=inter_top_val,
                               intra_top_val=intra_top_val,
                               diag_width=diag_width,
                               realSeqlen=realSeqlen)
            block_o = self.attention_exe._run_blocksparse(
                block_q, block_k, block_v, dense_mask)
            if self.attention_exe.attention_layout == "bnsh":
                if self.sparse_config.context_length:
                    ReorderBackTensorTextBNSH(res_o, block_o, head_ids,
                                              self.sparse_config.seqlen3d,
                                              self.sparse_config.block_size_q,
                                              self.sparse_config.num_blocks_q)
                else:
                    ReorderBackTensorBNSH(res_o, block_o, head_ids,
                                          self.sparse_config.seqlen3d,
                                          self.sparse_config.block_size_q,
                                          self.sparse_config.num_blocks_q)
            else:
                if self.sparse_config.context_length:
                    ReorderBackTensorTextBSNH(res_o, block_o, head_ids,
                                              self.sparse_config.seqlen3d,
                                              self.sparse_config.block_size_q,
                                              self.sparse_config.num_blocks_q)
                else:
                    ReorderBackTensorBSNH(res_o, block_o, head_ids,
                                          self.sparse_config.seqlen3d,
                                          self.sparse_config.block_size_q,
                                          self.sparse_config.num_blocks_q)

        summarizer_time = 0
        estimator_time = 0
        bsa_time = 0
        reorder_time = 0

        if q.device.type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(repeat):
                # Time summarizer
                start_event.record()
                summarize_res_dict = self.summarizer_exe(
                    q, k, v, head_ids, block_q, block_k, block_v, realSeqlen)
                end_event.record()
                torch.cuda.synchronize()
                summarizer_time += start_event.elapsed_time(end_event)

                # Time estimator
                start_event.record()
                self.estimator_exe(**summarize_res_dict,
                                   mask=dense_mask,
                                   inter_top_val=inter_top_val,
                                   intra_top_val=intra_top_val,
                                   diag_width=diag_width,
                                   realSeqlen=realSeqlen)
                end_event.record()
                torch.cuda.synchronize()
                estimator_time += start_event.elapsed_time(end_event)

                # Time attention
                start_event.record()
                block_o = self.attention_exe._run_blocksparse(
                    block_q, block_k, block_v, dense_mask)
                end_event.record()
                torch.cuda.synchronize()
                bsa_time += start_event.elapsed_time(end_event)

                start_event.record()
                if self.attention_exe.attention_layout == "bnsh":
                    if self.sparse_config.context_length:
                        ReorderBackTensorTextBNSH(
                            res_o, block_o, head_ids,
                            self.sparse_config.seqlen3d,
                            self.sparse_config.block_size_q,
                            self.sparse_config.num_blocks_q)
                    else:
                        ReorderBackTensorBNSH(res_o, block_o, head_ids,
                                              self.sparse_config.seqlen3d,
                                              self.sparse_config.block_size_q,
                                              self.sparse_config.num_blocks_q)
                else:
                    if self.sparse_config.context_length:
                        ReorderBackTensorTextBSNH(
                            res_o, block_o, head_ids,
                            self.sparse_config.seqlen3d,
                            self.sparse_config.block_size_q,
                            self.sparse_config.num_blocks_q)
                    else:
                        ReorderBackTensorBSNH(res_o, block_o, head_ids,
                                              self.sparse_config.seqlen3d,
                                              self.sparse_config.block_size_q,
                                              self.sparse_config.num_blocks_q)
                end_event.record()
                torch.cuda.synchronize()
                reorder_time += start_event.elapsed_time(end_event)
        else:
            for _ in range(repeat):
                # Time summarizer
                start = time.time()
                summarize_res_dict = self.summarizer_exe(
                    q, k, v, head_ids, block_q, block_k, block_v, realSeqlen)
                summarizer_time += (time.time() - start) * 1000

                # Time estimator
                start = time.time()
                self.estimator_exe(**summarize_res_dict,
                                   mask=dense_mask,
                                   inter_top_val=inter_top_val,
                                   intra_top_val=intra_top_val,
                                   diag_width=diag_width,
                                   realSeqlen=realSeqlen)
                estimator_time += (time.time() - start) * 1000

                # Time attention
                start = time.time()
                block_o = self.attention_exe._run_blocksparse(
                    block_q, block_k, block_v, dense_mask)
                bsa_time += (time.time() - start) * 1000

                start = time.time()
                if self.attention_exe.attention_layout == "bnsh":
                    if self.sparse_config.context_length:
                        ReorderBackTensorTextBNSH(
                            res_o, block_o, head_ids,
                            self.sparse_config.seqlen3d,
                            self.sparse_config.block_size_q,
                            self.sparse_config.num_blocks_q)
                    else:
                        ReorderBackTensorBNSH(res_o, block_o, head_ids,
                                              self.sparse_config.seqlen3d,
                                              self.sparse_config.block_size_q,
                                              self.sparse_config.num_blocks_q)
                else:
                    if self.sparse_config.context_length:
                        ReorderBackTensorTextBSNH(
                            res_o, block_o, head_ids,
                            self.sparse_config.seqlen3d,
                            self.sparse_config.block_size_q,
                            self.sparse_config.num_blocks_q)
                    else:
                        ReorderBackTensorBSNH(res_o, block_o, head_ids,
                                              self.sparse_config.seqlen3d,
                                              self.sparse_config.block_size_q,
                                              self.sparse_config.num_blocks_q)
                reorder_time += (time.time() - start) * 1000

        return {
            "summarizer_ms": summarizer_time / repeat,
            "estimator_ms": estimator_time / repeat,
            "bsa_ms": bsa_time / repeat,
            "reorder_ms": reorder_time / repeat,
        }
