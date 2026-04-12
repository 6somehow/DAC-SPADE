import torch
from typing import List, Optional, Union
import os
from math import prod
import jinja2

from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.executor import SummarizerExecutor, EstimatorExecutor, AttentionExecutor, E2EExecutor
from spade.core.codegen.titcompiler import TITCompiler
from spade.core.lower.utils.reorder import ReorderTensorBSNH, ReorderTensorBNSH


class TritonSummarizerExecutor(SummarizerExecutor):

    def __init__(self, sparse_config: SparseHeadConfig,
                 layout: str = "bsnh") -> None:
        super().__init__(sparse_config, layout=layout)
        self.compiler = TITCompiler()
        template_dir = os.path.join(os.path.dirname(__file__), 'template')
        jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True)
        self.template = jinja_env.get_template('summarizer.j2')

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 head_ids: torch.Tensor, block_q: torch.Tensor,
                 block_k: torch.Tensor,
                 block_v: torch.Tensor,
                 realSeqlen: Optional[int] = None) -> List[torch.Tensor]:
        '''
        q: (bsz, seqlen, num_heads, hidden_dim)
        k: (bsz, seqlen, num_heads, hidden_dim)
        v: (bsz, seqlen, num_heads, hidden_dim)
        block_q: if layout == "bsnh" -> (bsz, qNumBlocks, qBlockSize, num_head_ids, hidden_dim)
                 else -> (bsz, num_head_ids, qNumBlocks, qBlockSize, hidden_dim)
        block_k: if layout == "bsnh" -> (bsz, kvNumBlocks, kvBlockSize, num_head_ids, hidden_dim)
                 else -> (bsz, num_head_ids, kvNumBlocks, kvBlockSize, hidden_dim)
        block_v: if layout == "bsnh" -> (bsz, kvNumBlocks, kvBlockSize, num_head_ids, hidden_dim)
                 else -> (bsz, num_head_ids, kvNumBlocks, kvBlockSize, hidden_dim)
            
        return:
           if self.inter_summarize_enable:
                return block_inter_q: dict(str,tensor(bsz, num_head_ids, qNumBlocks, hidden_dim)), dict(str,tensor(bsz, num_head_ids, kNumBlocks, hidden_dim))
            if self.intra_summarize_enable:
                return block_intra_q (bsz, num_head_ids, qNumBlocks), block_intra_k (bsz, num_head_ids, kNumBlocks)
            
        '''
        if self.layout == "bsnh":
            reorder_tensor = ReorderTensorBSNH
        elif self.layout == "bnsh":
            reorder_tensor = ReorderTensorBNSH
        else:
            raise ValueError(f"Unsupported layout '{self.layout}'")

        seqlen3d = self.sparse_config.seqlen3d
        padding_val = 0

        kvNumBlocks3d = self.sparse_config.num_blocks_kv
        kvBlockSize3d = self.sparse_config.block_size_kv

        attn_type = self.sparse_config.attn_dtype

        # V reordering is not part of the summarization kernel
        reorder_tensor(block_v, v, head_ids, padding_val, seqlen3d,
                       kvBlockSize3d, kvNumBlocks3d)
        res = {}

        tensor_configs = {
            'q': (q, block_q, self.sparse_config.num_blocks_q,
                  self.sparse_config.block_size_q,
                  self.sparse_config.q_inter_summarizer_mode,
                  self.sparse_config.q_intra_summarizer_mode),
            'k': (k, block_k, self.sparse_config.num_blocks_kv,
                  self.sparse_config.block_size_kv,
                  self.sparse_config.k_inter_summarizer_mode,
                  self.sparse_config.k_intra_summarizer_mode)
        }

        for name, (x, block_x, numBlocks3d, blockSize3d, inter_modes,
                   intra_mode) in tensor_configs.items():
            bsz, _, num_heads, head_dim = x.shape
            stride_k1 = x.stride(1)
            stride_k2 = x.stride(2)
            num_head_ids = head_ids.shape[0]
            num_blocks = numBlocks3d[0] * numBlocks3d[1] * numBlocks3d[2]

            kernel_args = [head_ids, x, block_x]
            # Prepare output tensors
            block_inter_x = {}
            if self.sparse_config.inter_summarize_enable:
                if 'max' in inter_modes.values():
                    block_inter_x['max'] = torch.empty(bsz,
                                                       num_head_ids,
                                                       num_blocks,
                                                       head_dim,
                                                       device=x.device,
                                                       dtype=attn_type)
                    kernel_args.append(block_inter_x['max'])
                if 'min' in inter_modes.values():
                    block_inter_x['min'] = torch.empty(bsz,
                                                       num_head_ids,
                                                       num_blocks,
                                                       head_dim,
                                                       device=x.device,
                                                       dtype=attn_type)
                    kernel_args.append(block_inter_x['min'])
                if 'mean' in inter_modes.values():
                    block_inter_x['mean'] = torch.empty(bsz,
                                                        num_head_ids,
                                                        num_blocks,
                                                        head_dim,
                                                        device=x.device,
                                                        dtype=attn_type)
                    kernel_args.append(block_inter_x['mean'])
            else:
                inter_modes = {}
            block_intra_x = None
            if self.sparse_config.intra_summarize_enable:
                block_intra_x = torch.empty(bsz,
                                            num_head_ids,
                                            num_blocks,
                                            device=x.device,
                                            dtype=attn_type)
                kernel_args.append(block_intra_x)

            # Prepare context for jinja template
            context = {
                'INTER_SUMMARIZE_ENABLE':
                self.sparse_config.inter_summarize_enable,
                'INTER_SUMMARIZER_MODES': list(inter_modes.values()),
                'INTRA_SUMMARIZE_ENABLE':
                self.sparse_config.intra_summarize_enable,
                'stride_k0': x.stride(0),
                'stride_k1': stride_k1,
                'stride_k2': stride_k2,
                'stride_k3': x.stride(3),
                'stride_reorder_k0': block_x.stride(0),
                'stride_reorder_k1': block_x.stride(1),
                'stride_reorder_k2': block_x.stride(2),
                'stride_reorder_k3': block_x.stride(3),
                'stride_reorder_k4': block_x.stride(4),
                'layout_is_bsnh': 1 if self.layout == "bsnh" else 0,
                'frameDim': seqlen3d[0],
                'heightDim': seqlen3d[1],
                'widthDim': seqlen3d[2],
                'frameBlockSize': blockSize3d[0],
                'heightBlockSize': blockSize3d[1],
                'widthBlockSize': blockSize3d[2],
                'numFrameBlock': numBlocks3d[0],
                'numHeightBlock': numBlocks3d[1],
                'numWidthBlock': numBlocks3d[2],
                'numHeads': num_heads,
                'headDim': head_dim,
                'padding_val': padding_val,
            }
            if self.sparse_config.inter_summarize_enable and any(
                    inter_modes.values()):
                ref_tensor = next(iter(block_inter_x.values()))
                context.update({
                    'stride_layer0': ref_tensor.stride(0),
                    'stride_layer1': ref_tensor.stride(1),
                    'stride_layer2': ref_tensor.stride(2),
                    'stride_layer3': ref_tensor.stride(3)
                })
            if self.sparse_config.intra_summarize_enable:
                raise NotImplementedError("intra summarizer not implemented")
                context.update({
                    'stride_blockIntra0': block_intra_x.stride(0),
                    'stride_blockIntra1': block_intra_x.stride(1),
                    'stride_blockIntra2': block_intra_x.stride(2)
                })

            kernel_code = self.template.render(context)
            summarize_kernel = self.compiler.compile_kernel(
                kernel_code, 'summarize_kernel')

            grid = (bsz, num_head_ids, num_blocks)
            summarize_kernel[grid](*kernel_args)

            # Post-processing
            final_block_inter_x = {}
            if self.sparse_config.inter_summarize_enable:
                for mode_name, mode_val in inter_modes.items():
                    if mode_val == 'max':
                        final_block_inter_x[mode_name] = block_inter_x['max']
                    elif mode_val == 'min':
                        final_block_inter_x[mode_name] = block_inter_x['min']
                    elif mode_val == 'mean':
                        final_block_inter_x[mode_name] = block_inter_x['mean']

            res[f'block_inter_{name}'] = final_block_inter_x if self.sparse_config.inter_summarize_enable else None
            res[f'block_intra_{name}'] = block_intra_x if self.sparse_config.intra_summarize_enable else None

        return res


class TritonE2EExecutor(E2EExecutor):

    def __init__(self,
                 sparse_config: SparseHeadConfig,
                 layout: str = "bsnh",
                 backend: Optional[str] = None) -> None:
        self.summarizer_exe = TritonSummarizerExecutor(sparse_config,
                                                       layout=layout)
        self.estimator_exe = EstimatorExecutor(sparse_config)
        self.attention_exe = AttentionExecutor(sparse_config,
                                               layout=layout,
                                               backend=backend)
        super().__init__(sparse_config, self.summarizer_exe,
                         self.estimator_exe, self.attention_exe)

    def _check(self, tq: torch.Tensor) -> bool:
        if prod(self.sparse_config.seqlen3d) != tq.shape[1]:
            return False
        return True
