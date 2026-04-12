#include <torch/extension.h>
#include <vector>

at::Tensor cossim(at::Tensor k, std::vector<int64_t> seqlen3d,
                  std::vector<int64_t> block_shape,
                  std::vector<int64_t> num_blocks_on_axis,
                  int text_length = 0);

void mask_to_bsr(const torch::Tensor &sparse_mask, torch::Tensor &bsr,
                 torch::Tensor &num_blocks);

torch::Tensor scatter(torch::Tensor mask, torch::Tensor index,
                      torch::Tensor topk_size);

torch::Tensor static_sink_diag_set(torch::Tensor mask,
                                   torch::Tensor diag_mask_width,
                                   int sink_mask_width);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cossim", &cossim,
        "Calculate cosine similarity between token pairs within blocks.");
  m.def("mask_to_bsr", &mask_to_bsr, "Convert a sparse mask to BSR format.");
  m.def("scatter_mask", &scatter,
        "Scatter values into a mask tensor based on indices.");
  m.def("static_sink_diag_set", &static_sink_diag_set,
        "Set sink and diagonal blocks in a static mask.");
}
