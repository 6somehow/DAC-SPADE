import torch
from typing import List, Union, Optional
from math import prod
import os
import jinja2

from spade.core.lower.config import SparseHeadConfig
from spade.core.lower.executor import SummarizerExecutor
from spade.core.codegen.cudatcompiler import CUDATCompiler
from spade.core.lower.utils.reorder import (
    ReorderTensorBSNH,
    ReorderTensorBNSH,
    ReorderTensorTextBSNH,
    ReorderTensorTextBNSH,
)
from spade.core.lower.executor import E2EExecutor, EstimatorExecutor, AttentionExecutor


class CUDASummarizerExecutor(SummarizerExecutor):

    def __init__(self, sparse_config: SparseHeadConfig,
                 layout: str = "bsnh") -> None:
        super().__init__(sparse_config, layout=layout)
        self.compiler = CUDATCompiler()
        template_dir = os.path.join(os.path.dirname(__file__), 'template')
        jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True)
        # Point to the CUDA Jinja2 template
        self.template = jinja_env.get_template('summarizer_hdim_128.j2')

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 head_ids: torch.Tensor, block_q: torch.Tensor,
                 block_k: torch.Tensor, block_v: torch.Tensor,
                 realSeqlen: Optional[int]) -> List[torch.Tensor]:
        """
        Executes the summarization step using a compiled CUDA kernel.
        """
        if self.layout == "bsnh":
            reorder_tensor = ReorderTensorBSNH
            reorder_tensor_text = ReorderTensorTextBSNH
        elif self.layout == "bnsh":
            reorder_tensor = ReorderTensorBNSH
            reorder_tensor_text = ReorderTensorTextBNSH
        else:
            raise ValueError(f"Unsupported layout '{self.layout}'")

        seqlen3d = self.sparse_config.seqlen3d
        padding_val = 0.0

        kvNumBlocks3d = self.sparse_config.num_blocks_kv
        kvBlockSize3d = self.sparse_config.block_size_kv

        if self.sparse_config.context_length:
            reorder_tensor_text(block_v, v, head_ids, padding_val, seqlen3d,
                                kvBlockSize3d, kvNumBlocks3d, realSeqlen)
        else:
            reorder_tensor(block_v, v, head_ids, padding_val, seqlen3d,
                           kvBlockSize3d, kvNumBlocks3d)
        res = {}
        if self.sparse_config.dync_enable:
            tensor_configs = {
                'q': (q, block_q, self.sparse_config.num_blocks_q,
                      self.sparse_config.block_size_q,
                      self.sparse_config.q_inter_summarizer_mode,
                      self.sparse_config.q_intra_summarizer_mode,
                      getattr(self.sparse_config, 'context_q_num_block',
                              None)),
                'k': (k, block_k, self.sparse_config.num_blocks_kv,
                      self.sparse_config.block_size_kv,
                      self.sparse_config.k_inter_summarizer_mode,
                      self.sparse_config.k_intra_summarizer_mode,
                      getattr(self.sparse_config, 'context_kv_num_block',
                              None))
            }

            for name, (x, block_x, numBlocks3d, blockSize3d, inter_modes,
                       intra_mode, context_blocks) in tensor_configs.items():
                       
                bsz, _, num_heads, head_dim = x.shape
                num_head_ids = head_ids.shape[0]
                num_blocks = prod(numBlocks3d) + (context_blocks
                                                  if context_blocks else 0)

                # The kernel needs a flat list of modes like ['max', 'min']
                inter_modes_list = list(inter_modes.values())

                kernel_args = [head_ids, x, block_x]
                block_inter_x = {}

                # Prepare output tensors for inter-block summarization
                # The order must match the Jinja template's function signature
                if 'max' in inter_modes_list:
                    block_inter_x['max'] = torch.empty(
                        bsz,
                        num_head_ids,
                        num_blocks,
                        head_dim,
                        device=x.device,
                        dtype=self.sparse_config.attn_dtype)
                    kernel_args.append(block_inter_x['max'])
                if 'min' in inter_modes_list:
                    block_inter_x['min'] = torch.empty(
                        bsz,
                        num_head_ids,
                        num_blocks,
                        head_dim,
                        device=x.device,
                        dtype=self.sparse_config.attn_dtype)
                    kernel_args.append(block_inter_x['min'])
                if 'mean' in inter_modes_list:
                    block_inter_x['mean'] = torch.empty(
                        bsz,
                        num_head_ids,
                        num_blocks,
                        head_dim,
                        device=x.device,
                        dtype=self.sparse_config.attn_dtype)
                    kernel_args.append(block_inter_x['mean'])

                # Prepare output tensor for intra-block summarization
                block_intra_x = None
                if self.sparse_config.intra_summarize_enable:
                    block_intra_x = torch.empty(
                        bsz,
                        num_head_ids,
                        num_blocks,
                        device=x.device,
                        dtype=self.sparse_config.attn_dtype)
                    kernel_args.append(block_intra_x)

                # Add realSeqlen if text processing is enabled
                if self.sparse_config.context_length:
                    kernel_args.append(realSeqlen)

                # Map PyTorch dtype to the string expected by the Jinja template
                dtype_str = 'bfloat16' if self.sparse_config.attn_dtype == torch.bfloat16 else 'float16'

                block_size = blockSize3d[0] * blockSize3d[1] * blockSize3d[2]
                num_context_blocks = None
                if self.sparse_config.context_length and block_size > 0:
                    num_context_blocks = (self.sparse_config.context_length +
                                          block_size - 1) // block_size

                # Prepare context for the Jinja template
                context = {
                    'DTYPE': dtype_str,
                    'LAYOUT': self.layout,
                    'INTER_SUMMARIZER_MODES': inter_modes_list,
                    'INTRA_SUMMARIZE_ENABLE':
                    self.sparse_config.intra_summarize_enable,
                    'CONTEXT_LEN': self.sparse_config.context_length,
                    'NUMCONTEXTBLOCK': num_context_blocks,
                    'num_heads': num_heads,
                    'frameDim': seqlen3d[0],
                    'heightDim': seqlen3d[1],
                    'widthDim': seqlen3d[2],
                    'frameBlockSize': blockSize3d[0],
                    'heightBlockSize': blockSize3d[1],
                    'widthBlockSize': blockSize3d[2],
                    'numFrameBlock': numBlocks3d[0],
                    'numHeightBlock': numBlocks3d[1],
                    'numWidthBlock': numBlocks3d[2],
                    'headDim': head_dim,
                    'padding_val': padding_val,
                }

                kernel_code = self.template.render(context)

                # Compile using CUDATCompiler and get the module
                module_name = f"summarize_{name}_{dtype_str}_{self.layout}_{'_'.join(sorted(inter_modes_list))}"
                summarize_module = self.compiler.compile_module(
                    kernel_code, module_name)

                contiguous_args = [
                    arg.contiguous() if isinstance(arg, torch.Tensor) else arg
                    for arg in kernel_args
                ]
                summarize_module.summarize_forward(*contiguous_args)

                # Organize results into the final dictionary
                final_block_inter_x = {}
                for mode_name, mode_val in inter_modes.items():
                    if mode_val in block_inter_x:
                        final_block_inter_x[mode_name] = block_inter_x[
                            mode_val]

                res[f'block_inter_{name}'] = final_block_inter_x if any(
                    inter_modes_list) else None
                res[f'block_intra_{name}'] = block_intra_x if intra_mode else None
        else:
            # Handle the case where dynamic summarization is disabled
            res.update({
                'block_inter_q': None,
                'block_inter_k': None,
                'block_intra_q': None,
                'block_intra_k': None,
            })
            qNumBlocks3d = self.sparse_config.num_blocks_q
            qBlockSize3d = self.sparse_config.block_size_q
            reorder_tensor(block_q, q, head_ids, padding_val, seqlen3d,
                           qBlockSize3d, qNumBlocks3d)
            reorder_tensor(block_k, k, head_ids, padding_val, seqlen3d,
                           kvBlockSize3d, kvNumBlocks3d)

        return res


class CUDAE2EExecutor(E2EExecutor):

    def __init__(self, sparse_config: SparseHeadConfig) -> None:
        self.summarizer_exe = CUDASummarizerExecutor(sparse_config)
        self.estimator_exe = EstimatorExecutor(sparse_config)
        self.attention_exe = AttentionExecutor(sparse_config)
        super().__init__(sparse_config, self.summarizer_exe,
                         self.estimator_exe, self.attention_exe)

    def _check(self, tq: torch.Tensor) -> bool:
        if prod(self.sparse_config.seqlen3d) != tq.shape[1]:
            return False
        return True
