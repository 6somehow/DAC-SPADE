#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// Dtype specific includes
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// --- CUDA Constants ---
constexpr int WARP_SIZE = 32;
constexpr int VECTOR_SIZE = 4; // Each thread processes 4 elements (e.g., a float4)
constexpr float PADDING_VAL = 0.0f;

// --- [Type-Agnostic Helper Utilities] ---

// A traits struct to handle type-specific details for 16-bit float types.
template <typename T> struct CudaTypeTraits;

// Specialization for float16 (__half)
template <>
struct CudaTypeTraits<__half> {
    using VecType = __half2; // Vector type (__half2) handles 2 halfs at a time.
    // Convert from float2 to VecType
    static __device__ __forceinline__ VecType from_float2(const float2& f) { return __floats2half2_rn(f.x,f.y); }
    // Convert two VecTypes (4 elements total) to a float4 for computation
    static __device__ __forceinline__ __half from_float(float val) {
        return __float2half(val);}
    static __device__ __forceinline__ float4 to_float4(const VecType& a, const VecType& b) {
        float2 fa = __half22float2(a);
        float2 fb = __half22float2(b);
        return make_float4(fa.x, fa.y, fb.x, fb.y);
    }
};

// Specialization for bfloat16 (__nv_bfloat16)
template <>
struct CudaTypeTraits<__nv_bfloat16> {
    using VecType = __nv_bfloat162; // Vector type (__nv_bfloat162) handles 2 bfloats at a time.
    // Convert from float2 to VecType
    static __device__ __forceinline__ VecType from_float2(const float2& f) { return __floats2bfloat162_rn(f.x, f.y); }
    // Convert two VecTypes (4 elements total) to a float4 for computation
    static __device__ __forceinline__ __nv_bfloat16 from_float(float val) {
        return __float2bfloat16(val);
    }
    static __device__ __forceinline__ float4 to_float4(const VecType& a, const VecType& b) {
        float2 fa = __bfloat1622float2(a);
        float2 fb = __bfloat1622float2(b);
        return make_float4(fa.x, fa.y, fb.x, fb.y);
    }
};

// --- Vector Math Operators for float4 ---
__device__ __forceinline__ float4 operator*(const float4 &a, const float4 &b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float4 operator+(const float4 &a, const float4 &b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 operator*(const float4 &a, float b) { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
__device__ __forceinline__ float4 fmaxf(const float4 &a, const float4 &b) { return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w)); }
__device__ __forceinline__ float4 fminf(const float4 &a, const float4 &b) { return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w)); }

// Standard warp-level sum reduction
__inline__ __device__ float warpAllReduceSum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val; // The root thread (lane 0) is not needed, as all threads get the sum
}

// --- CUDA Device Functions ---

template <typename T>
__device__ __forceinline__ void pad_vector(T *reorder_x_ptr) {
    using traits = CudaTypeTraits<T>;
    using VecType = typename traits::VecType;
    auto* reorder_x_vec_ptr = reinterpret_cast<VecType*>(reorder_x_ptr);

    const float2 padding_f2 = make_float2(PADDING_VAL, PADDING_VAL);
    reorder_x_vec_ptr[0] = traits::from_float2(padding_f2);
    reorder_x_vec_ptr[1] = traits::from_float2(padding_f2);
}

template <typename T>
__device__ __forceinline__ void process_vector(
    const T *x_ptr, T *reorder_x_ptr,
    float4 &localMax4, float4 &localMin4, float4 &preNormSum4, float4 &cosSim4) {

    using traits = CudaTypeTraits<T>;
    using VecType = typename traits::VecType;
    const auto *x_vec_ptr = reinterpret_cast<const VecType *>(x_ptr);
    auto* reorder_x_vec_ptr = reinterpret_cast<VecType*>(reorder_x_ptr);

    // Reorder X: Copy 2 vector types (total of 4 elements)
    reorder_x_vec_ptr[0] = x_vec_ptr[0];
    reorder_x_vec_ptr[1] = x_vec_ptr[1];

    // Load and convert to float4 for computation
    float4 fval4 = traits::to_float4(x_vec_ptr[0], x_vec_ptr[1]);

    localMax4 = fmaxf(localMax4, fval4);
    localMin4 = fminf(localMin4, fval4);

    // Normalize vector across the warp for Cosine Similarity
    float normFactor_scalar = (fval4.x * fval4.x) + (fval4.y * fval4.y) +
                              (fval4.z * fval4.z) + (fval4.w * fval4.w);
    float total_norm_sq = warpAllReduceSum(normFactor_scalar);
    float norm = rsqrtf(total_norm_sq + 1e-8f);
    float4 normVal4 = fval4 * norm;

    // This computes sum of pairwise cosine similarities: sum_{i<j} (v_i . v_j)
    cosSim4 = cosSim4 + preNormSum4 * normVal4 * 2.0f;
    preNormSum4 = preNormSum4 + normVal4;
}

template <typename T>
__device__ __forceinline__ void store_results(
    int64_t bsz_idx, int64_t head_list_idx, int64_t block_idx_z,
    int64_t total_num_blocks, int64_t num_head_list, int64_t head_dim, int64_t thread_idx,
    int64_t num_elements,
    const float4 &localMax4, T *block_max,
    const float4 &localMin4, T *block_min,
    const float4 &cosSim4, T *block_cos_sim) {

    using traits = CudaTypeTraits<T>;
    using VecType = typename traits::VecType;

    int64_t layer_idx_base = bsz_idx * num_head_list * total_num_blocks * head_dim +
                             head_list_idx * total_num_blocks * head_dim +
                             block_idx_z * head_dim;

    // Store Max and Min
    T *block_max_ptr = block_max + layer_idx_base + thread_idx * VECTOR_SIZE;
    auto* block_max_vec_ptr = reinterpret_cast<VecType*>(block_max_ptr);
    block_max_vec_ptr[0] = traits::from_float2(make_float2(localMax4.x, localMax4.y));
    block_max_vec_ptr[1] = traits::from_float2(make_float2(localMax4.z, localMax4.w));

    T *block_min_ptr = block_min + layer_idx_base + thread_idx * VECTOR_SIZE;
    auto* block_min_vec_ptr = reinterpret_cast<VecType*>(block_min_ptr);
    block_min_vec_ptr[0] = traits::from_float2(make_float2(localMin4.x, localMin4.y));
    block_min_vec_ptr[1] = traits::from_float2(make_float2(localMin4.z, localMin4.w));

    // Reduce and store Cosine Similarity
    float cosSim_scalar = cosSim4.x + cosSim4.y + cosSim4.z + cosSim4.w;
    cosSim_scalar = warpAllReduceSum(cosSim_scalar);

    if (thread_idx == 0) {
        int64_t layerIdx = bsz_idx * num_head_list * total_num_blocks +
                           head_list_idx * total_num_blocks + block_idx_z;
        
        // --- FIX IS HERE ---
        // 1. Calculate the result as a standard float
        float result_float = (num_elements > 1)
            ? (cosSim_scalar / (num_elements * (num_elements - 1)))
            : 0.0f;
        
        // 2. Use the new type traits helper for correct conversion
        block_cos_sim[layerIdx] = traits::from_float(result_float);
    }
}
// --- CUDA Kernel ---

template <typename T>
__global__ void summarize_kernel(
    const int64_t* __restrict__ head_indices,
    const T* __restrict__ x,
    T* __restrict__ reorder_x,
    T* __restrict__ block_max,
    T* __restrict__ block_min,
    T* __restrict__ block_cos_sim,
    const int64_t num_head_list,
    const int64_t real_seqlen,
    // Dimension parameters
    const int64_t frame_dim, const int64_t height_dim, const int64_t width_dim,
    const int64_t context_len,
    const int64_t frame_block_size, const int64_t height_block_size, const int64_t width_block_size,
    const int64_t num_frame_block, const int64_t num_height_block, const int64_t num_width_block,
    const int64_t num_context_block,
    const int64_t num_heads, const int64_t head_dim) {

    // --- Setup and Indices ---
    const int64_t bsz_idx = blockIdx.x;
    const int64_t head_list_idx = blockIdx.y;
    const int64_t block_idx_z = blockIdx.z;
    const int64_t thread_idx = threadIdx.x;

    const int64_t num_video_blocks = num_frame_block * num_height_block * num_width_block;
    const int64_t block_size = frame_block_size * height_block_size * width_block_size;
    const int64_t video_seq_len = frame_dim * height_dim * width_dim;
    const int64_t total_x_seq_len = video_seq_len + context_len;
    const int64_t total_num_blocks = num_video_blocks + num_context_block;

    const int64_t head_idx = head_indices[head_list_idx];
    if (head_idx < 0 || head_idx >= num_heads) return; // Safeguard

    // --- Main Logic: Differentiate Video and Text Blocks ---
    if (block_idx_z < num_video_blocks) {
        // --- [Video Block Processing] ---
        float4 localMax4 = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
        float4 localMin4 = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
        float4 preNormSum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 cosSim4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int64_t num_elements = 0;

        const int64_t plane_blocks = num_height_block * num_width_block;
        const int64_t frame_block_idx = block_idx_z / plane_blocks;
        const int64_t height_block_idx = (block_idx_z % plane_blocks) / num_width_block;
        const int64_t width_block_idx = (block_idx_z % plane_blocks) % num_width_block;

        for (int64_t fb = 0; fb < frame_block_size; ++fb) {
            const int64_t frame_idx = fb + frame_block_idx * frame_block_size;
            for (int64_t hb = 0; hb < height_block_size; ++hb) {
                const int64_t height_idx = hb + height_block_idx * height_block_size;
                for (int64_t wb = 0; wb < width_block_size; ++wb) {
                    const int64_t width_idx = wb + width_block_idx * width_block_size;
                    const int64_t b_block_idx = fb * height_block_size * width_block_size + hb * width_block_size + wb;
                    const int64_t new_x_idx_base =
                        bsz_idx * num_head_list * total_num_blocks * block_size * head_dim +
                        head_list_idx * total_num_blocks * block_size * head_dim +
                        block_idx_z * block_size * head_dim + b_block_idx * head_dim;
                    T *reorder_x_ptr = reorder_x + new_x_idx_base + thread_idx * VECTOR_SIZE;

                    if (frame_idx < frame_dim && height_idx < height_dim && width_idx < width_dim) {
                        num_elements++;
                        const int64_t seq_idx = frame_idx * height_dim * width_dim + height_idx * width_dim + width_idx;
                        const int64_t x_idx_base = bsz_idx * total_x_seq_len * num_heads * head_dim +
                            seq_idx * num_heads * head_dim + head_idx * head_dim;
                        const T *x_ptr = x + x_idx_base + thread_idx * VECTOR_SIZE;
                        process_vector<T>(x_ptr, reorder_x_ptr, localMax4, localMin4, preNormSum4, cosSim4);
                    } else {
                        pad_vector<T>(reorder_x_ptr);
                    }
                }
            }
        }
        store_results<T>(bsz_idx, head_list_idx, block_idx_z, total_num_blocks, num_head_list, head_dim, thread_idx, num_elements,
                         localMax4, block_max, localMin4, block_min, cosSim4, block_cos_sim);
    } else {
        // --- [Text Block Processing] ---
        const int64_t t_block_idx = block_idx_z - num_video_blocks;
        for (int64_t i = 0; i < block_size; ++i) {
            const int64_t seq_idx = video_seq_len + t_block_idx * block_size + i;
            const int64_t new_x_idx_base =
                bsz_idx * num_head_list * total_num_blocks * block_size * head_dim +
                head_list_idx * total_num_blocks * block_size * head_dim +
                block_idx_z * block_size * head_dim + i * head_dim;
            T *reorder_x_ptr = reorder_x + new_x_idx_base + thread_idx * VECTOR_SIZE;

            if (seq_idx < real_seqlen) {
                const int64_t x_idx_base = bsz_idx * total_x_seq_len * num_heads * head_dim +
                    seq_idx * num_heads * head_dim + head_idx * head_dim;
                const T *x_ptr = x + x_idx_base + thread_idx * VECTOR_SIZE;
                
                using VecType = typename CudaTypeTraits<T>::VecType;
                reinterpret_cast<VecType *>(reorder_x_ptr)[0] = reinterpret_cast<const VecType *>(x_ptr)[0];
                reinterpret_cast<VecType *>(reorder_x_ptr)[1] = reinterpret_cast<const VecType *>(x_ptr)[1];
            } else {
                pad_vector<T>(reorder_x_ptr);
            }
        }
    }
}

// --- C++ Wrapper Function ---
void summarize_forward(
    at::Tensor head_indices,
    at::Tensor x,
    at::Tensor reorder_x,
    at::Tensor block_max,
    at::Tensor block_min,
    at::Tensor block_cos_sim,
    int64_t real_seqlen,
    // Dimension args
    int64_t frame_dim, int64_t height_dim, int64_t width_dim,
    int64_t context_len,
    int64_t frame_block_size, int64_t height_block_size, int64_t width_block_size,
    int64_t num_context_block) {
    // --- Input Validation ---
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    // Check all other tensors for CUDA device and contiguity
    TORCH_CHECK(head_indices.device().is_cuda() && head_indices.is_contiguous(), "head_indices must be a contiguous CUDA tensor");
    TORCH_CHECK(reorder_x.is_contiguous() && reorder_x.device().is_cuda(), "reorder_x must be a contiguous CUDA tensor");
    TORCH_CHECK(block_max.is_contiguous() && block_max.device().is_cuda(), "block_max must be a contiguous CUDA tensor");
    TORCH_CHECK(block_min.is_contiguous() && block_min.device().is_cuda(), "block_min must be a contiguous CUDA tensor");
    TORCH_CHECK(block_cos_sim.is_contiguous() && block_cos_sim.device().is_cuda(), "block_cos_sim must be a contiguous CUDA tensor");

    // --- Dimension Calculations ---
    const int64_t bsz = x.size(0);
    const int64_t num_heads = x.size(2);
    const int64_t head_dim = x.size(3);
    const int64_t num_head_list = head_indices.size(0);

    TORCH_CHECK(head_dim % (WARP_SIZE * VECTOR_SIZE) == 0, "HEADDIM must be a multiple of WARP_SIZE * VECTOR_SIZE");

    auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    const int64_t num_frame_block = ceil_div(frame_dim, frame_block_size);
    const int64_t num_height_block = ceil_div(height_dim, height_block_size);
    const int64_t num_width_block = ceil_div(width_dim, width_block_size);
    const int64_t num_video_blocks = num_frame_block * num_height_block * num_width_block;
    const int64_t num_blocks_z = num_video_blocks + num_context_block;

    // --- Kernel Launch Configuration ---
    dim3 grid(bsz, num_head_list, num_blocks_z);
    dim3 block(head_dim / VECTOR_SIZE);
    TORCH_CHECK(block.x == WARP_SIZE, "Number of threads per block must equal WARP_SIZE");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == at::ScalarType::Half) {
        summarize_kernel<__half><<<grid, block, 0, stream>>>(
            head_indices.data_ptr<int64_t>(),
            reinterpret_cast<const __half*>(x.data_ptr()),
            reinterpret_cast<__half*>(reorder_x.data_ptr()),
            reinterpret_cast<__half*>(block_max.data_ptr()),
            reinterpret_cast<__half*>(block_min.data_ptr()),
            reinterpret_cast<__half*>(block_cos_sim.data_ptr()),
            num_head_list, real_seqlen,
            frame_dim, height_dim, width_dim, context_len,
            frame_block_size, height_block_size, width_block_size,
            num_frame_block, num_height_block, num_width_block,
            num_context_block, num_heads, head_dim);
    } else if (x.scalar_type() == at::ScalarType::BFloat16) {
        summarize_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            head_indices.data_ptr<int64_t>(),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(reorder_x.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(block_max.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(block_min.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(block_cos_sim.data_ptr()),
            num_head_list, real_seqlen,
            frame_dim, height_dim, width_dim, context_len,
            frame_block_size, height_block_size, width_block_size,
            num_frame_block, num_height_block, num_width_block,
            num_context_block, num_heads, head_dim);
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only float16 and bfloat16 are supported.");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --- PYBIND11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("summarize_forward", &summarize_forward, "Summarizer forward (CUDA, fp16/bf16 only)",
        py::arg("head_indices"),
        py::arg("x"),
        py::arg("reorder_x"),
        py::arg("block_max"),
        py::arg("block_min"),
        py::arg("block_cos_sim"),
        py::arg("real_seqlen"),
        py::arg("frame_dim"),
        py::arg("height_dim"),
        py::arg("width_dim"),
        py::arg("context_len"),
        py::arg("frame_block_size"),
        py::arg("height_block_size"),
        py::arg("width_block_size"),
        py::arg("num_context_block")
    );
}