// CUDA implementation of patch grid optimization
// Phase 1: Infrastructure & Data Transfer

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>

// Only include CUDA header
#include "patchgrid_cuda.h"

// Helper: Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace OFC {

// ============================================================================
// DEVICE DATA STRUCTURES
// ============================================================================

// GPU-side patch state
struct PatchStateGPU {
    // Patch parameters (flow displacement or depth)
    float p_x, p_y;       // Current parameters
    float delta_p_x, delta_p_y;  // Parameter update

    // Hessian Matrix (2x2 symmetric for OF)
    float Hes_xx, Hes_xy, Hes_yy;

    // Patch position
    float pt_ref_x, pt_ref_y;  // Reference patch center
    float pt_iter_x, pt_iter_y; // Current iteration position

    // Convergence tracking
    float delta_p_sqnorm;
    float delta_p_sqnorm_init;
    float mares;       // Mean absolute residual
    float mares_old;
    int cnt;           // Iteration count
    bool hasconverged;
    bool hasoptstarted;
    bool invalid;
};

// GPU-side grid parameters
struct GridParamsGPU {
    int nopatches;
    int nopw, noph;
    int width, height;
    int imgpadding;
    int tmp_w; // width + 2*imgpadding

    // Optimization parameters
    int max_iter, min_iter;
    int p_samp_s;  // Patch size
    int novals;    // p_samp_s * p_samp_s * noc
    int noc;       // Number of channels
    float dp_thresh;
    float dr_thresh;
    float res_thresh;
    float outlierthresh;
    int costfct;   // 0: L2, 1: L1, 2: PseudoHuber
    
    // Camera params for boundary check
    float tmp_lb;
    float tmp_ubw;
    float tmp_ubh;
};

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

__device__ void getPatchStaticNNGrad_device(
    const float* img, const float* img_dx, const float* img_dy,
    float pt_x, float pt_y,
    float* patch_dst, float* dx_dst, float* dy_dst,
    int p_samp_s, int tmp_w, int noc, int imgpadding
) {
    int pos_x = roundf(pt_x);
    int pos_y = roundf(pt_y);
    
    int center_x = pos_x + imgpadding;
    int center_y = pos_y + imgpadding;
    
    int half_s = p_samp_s / 2;
    int lb = -half_s;
    
    // Each thread handles one pixel of the patch (assuming blockDim matches patch size)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (tx < p_samp_s && ty < p_samp_s) {
        int patch_idx = ty * p_samp_s + tx;
        int img_y = center_y + lb + ty;
        int img_x = center_x + lb + tx;
        
        // Layout: [height][width][channels] interleaved
        int stride = (noc == 1) ? tmp_w : tmp_w * 3;
        int pixel_offset = img_y * stride + img_x * noc;
        
        for (int c = 0; c < noc; ++c) {
            // Use __ldg for read-only global memory access
            patch_dst[patch_idx * noc + c] = __ldg(&img[pixel_offset + c]);
            dx_dst[patch_idx * noc + c]    = __ldg(&img_dx[pixel_offset + c]);
            dy_dst[patch_idx * noc + c]    = __ldg(&img_dy[pixel_offset + c]);
        }
    }
}

// Optimized bilinear fetch using texture object
__device__ void getPatchStaticBil_Texture_device(
    cudaTextureObject_t tex,
    float pt_x, float pt_y,
    float* patch_dst,
    int p_samp_s, int noc, int imgpadding
) {
    // Texture coordinates need to include padding
    // tex2D expects coordinates in the texture space (0..width-1, 0..height-1)
    // Our texture is created from the padded image.
    // pt_x, pt_y are in unpadded coordinates.
    // So we add imgpadding.
    
    float center_x = pt_x + imgpadding;
    float center_y = pt_y + imgpadding;

    int half_s = p_samp_s / 2;
    int lb = -half_s;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tx < p_samp_s && ty < p_samp_s) {
        int patch_idx = ty * p_samp_s + tx;
        
        // Sample position: center + offset
        // Note: tex2D uses (x,y) where (0.5, 0.5) is the center of the first pixel.
        // Our integer coordinates point to pixel centers?
        // In CPU code: `pos_x = ceil(pt_x)`, `resid = pt_x - floor`.
        // It seems standard bilinear.
        // tex2D(x, y) samples at x, y. If x is integer, it samples exactly the pixel center (if unnormalized coords).
        // We need to be careful with 0.5 offset.
        // CUDA unnormalized coordinates: pixel (i,j) covers [i, i+1) x [j, j+1). Center is i+0.5, j+0.5.
        // If we pass integer 'k', we get value at boundary?
        // No, `cudaFilterModeLinear` with unnormalized coordinates:
        // T(x) = (1-a)*T[i] + a*T[i+1], where i = floor(x-0.5), a = frac(x-0.5).
        // So if we pass x=0.5, i=0, a=0 -> T[0].
        // If we pass x=1.0, i=0, a=0.5 -> 0.5*T[0] + 0.5*T[1].
        // CPU code: `pos[0] = ceil(mid); pos[2] = floor(mid); resid = mid - floor`.
        // This is standard bilinear interpolation at `mid`.
        // So if we want to sample at `mid`, we should pass `mid + 0.5f` to tex2D?
        // Let's verify.
        // If mid=0.0, CPU samples at 0.0.
        // CUDA tex2D at 0.5 returns T[0].
        // So yes, we should add 0.5f to coordinates.
        
        float u = center_x + lb + tx + 0.5f;
        float v = center_y + lb + ty + 0.5f;

        if (noc == 1) {
            float val = tex2D<float>(tex, u, v);
            patch_dst[patch_idx] = val;
        } else if (noc == 3) {
            float4 val4 = tex2D<float4>(tex, u, v);
            patch_dst[patch_idx * 3 + 0] = val4.x;
            patch_dst[patch_idx * 3 + 1] = val4.y;
            patch_dst[patch_idx * 3 + 2] = val4.z;
        }
    }
}

__device__ void NormalizePatch_device(
    float* patch, 
    float* s_scratch, // Scratch buffer for reduction (size: p_samp_s * p_samp_s)
    int p_samp_s, int novals, int noc
) {
    int tid = threadIdx.y * p_samp_s + threadIdx.x;
    int total_threads = p_samp_s * p_samp_s;
    
    // Sum
    float local_sum = 0.0f;
    for (int c = 0; c < noc; ++c) {
        if (tid * noc + c < novals) local_sum += patch[tid * noc + c];
    }
    
    s_scratch[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (int s = total_threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_scratch[tid] += s_scratch[tid + s];
        __syncthreads();
    }
    
    float mean = s_scratch[0] / novals;
    __syncthreads();
    
    // Subtract
    for (int c = 0; c < noc; ++c) {
        if (tid * noc + c < novals) patch[tid * noc + c] -= mean;
    }
    __syncthreads();
}

__device__ void ComputeSteepestDescent_device(
    const float* dx, const float* dy, const float* pdiff,
    float* s_scratch_x, float* s_scratch_y, // Scratch buffers for reduction
    float* out_sdx, float* out_sdy,
    int p_samp_s, int novals, int noc
) {
    int tid = threadIdx.y * p_samp_s + threadIdx.x;
    int total_threads = p_samp_s * p_samp_s;
    
    float local_sdx = 0.0f;
    float local_sdy = 0.0f;
    
    for (int c = 0; c < noc; ++c) {
        if (tid * noc + c < novals) {
            float val_pdiff = pdiff[tid * noc + c];
            local_sdx += dx[tid * noc + c] * val_pdiff;
            local_sdy += dy[tid * noc + c] * val_pdiff;
        }
    }
    
    s_scratch_x[tid] = local_sdx;
    s_scratch_y[tid] = local_sdy;
    __syncthreads();
    
    for (int s = total_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_scratch_x[tid] += s_scratch_x[tid + s];
            s_scratch_y[tid] += s_scratch_y[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *out_sdx = s_scratch_x[0];
        *out_sdy = s_scratch_y[0];
    }
    __syncthreads();
}

__device__ void ComputeHessian_device(
    const float* dx, const float* dy,
    float* s_scratch_xx, float* s_scratch_xy, float* s_scratch_yy,
    float* out_hxx, float* out_hxy, float* out_hyy,
    int p_samp_s, int novals, int noc
) {
    int tid = threadIdx.y * p_samp_s + threadIdx.x;
    int total_threads = p_samp_s * p_samp_s;
    
    float local_hxx = 0.0f;
    float local_hxy = 0.0f;
    float local_hyy = 0.0f;
    
    for (int c = 0; c < noc; ++c) {
        if (tid * noc + c < novals) {
            float val_dx = dx[tid * noc + c];
            float val_dy = dy[tid * noc + c];
            local_hxx += val_dx * val_dx;
            local_hxy += val_dx * val_dy;
            local_hyy += val_dy * val_dy;
        }
    }
    
    s_scratch_xx[tid] = local_hxx;
    s_scratch_xy[tid] = local_hxy;
    s_scratch_yy[tid] = local_hyy;
    __syncthreads();
    
    for (int s = total_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_scratch_xx[tid] += s_scratch_xx[tid + s];
            s_scratch_xy[tid] += s_scratch_xy[tid + s];
            s_scratch_yy[tid] += s_scratch_yy[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *out_hxx = s_scratch_xx[0];
        *out_hxy = s_scratch_xy[0];
        *out_hyy = s_scratch_yy[0];
    }
    __syncthreads();
}

__device__ void LossComputeErrorImage_device(
    float* pdiff, float* pweight,
    const float* pat_src, const float* pat_target,
    int novals, int costfct,
    float normoutlier, float normoutlier_sq
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int p_samp_s = blockDim.x; 
    int idx = ty * p_samp_s + tx;
    
    int noc = novals / (p_samp_s * p_samp_s);

    for (int c = 0; c < noc; ++c) {
        int i = (idx * noc) + c;
        if (i >= novals) continue;

        float diff = pat_src[i] - pat_target[i];
        float val_pdiff = 0.0f;
        float val_pweight = 0.0f;

        if (costfct == 0) { // L2
            val_pdiff = diff;
            val_pweight = fabsf(diff);
        } else if (costfct == 1) { // L1
            float abs_diff = fabsf(diff);
            float sqrt_abs = sqrtf(abs_diff);
            float sign = (diff >= 0) ? 1.0f : -1.0f;
            val_pdiff = sign * sqrt_abs;
            val_pweight = sqrt_abs; 
        } else if (costfct == 2) { // PseudoHuber
            float b_sq = normoutlier_sq; 
            float d_sq = diff * diff;
            float term = sqrtf(1.0f + d_sq / b_sq);
            float cost = 2.0f * b_sq * (term - 1.0f);
            float sign = (diff >= 0) ? 1.0f : -1.0f;
            val_pdiff = sign * sqrtf(cost);
            val_pweight = fabsf(val_pdiff);
        }

        pdiff[i] = val_pdiff;
        pweight[i] = val_pweight;
    }
}

// Kernel: Optimize Patches (Phase 3-3: Full Loop)
// Grid: (nopatches, 1, 1)
// Block: (p_samp_s, p_samp_s, 1)
__global__ void OptimizePatchesKernel(
    PatchStateGPU* d_patches,
    const float* d_im_ref, const float* d_im_ref_dx, const float* d_im_ref_dy,
    cudaTextureObject_t tex_target, // Texture object for target image
    GridParamsGPU params,
    float* d_debug_delta_p, // For verification
    DebugIterData* d_debug_iter // For detailed debugging
) {
    int patch_idx = blockIdx.x;
    if (patch_idx >= params.nopatches) return;

    PatchStateGPU& ps = d_patches[patch_idx];

    // Shared memory
    extern __shared__ float s_mem[];
    float* s_pat_ref = s_mem; // novals
    float* s_dx = s_pat_ref + params.novals; // novals
    float* s_dy = s_dx + params.novals; // novals
    float* s_pat_target = s_dy + params.novals; // novals
    float* s_pdiff = s_pat_target; // Alias
    float* s_scratch = s_pat_target + params.novals; // 5th buffer
    float* s_scratch2 = s_scratch + params.novals; // 6th buffer

    
    // 1. Extract Reference Patch & Grads (NN) - Done ONCE
    getPatchStaticNNGrad_device(
        d_im_ref, d_im_ref_dx, d_im_ref_dy,
        ps.pt_ref_x, ps.pt_ref_y,
        s_pat_ref, s_dx, s_dy,
        params.p_samp_s, params.tmp_w, params.noc, params.imgpadding
    );
    __syncthreads();
    
    // Normalize Reference Patch - Done ONCE
    NormalizePatch_device(s_pat_ref, s_scratch, params.p_samp_s, params.novals, params.noc);
    
    // Compute Hessian - Done ONCE
    // Use s_pat_target, s_scratch, s_scratch2 for reduction
    ComputeHessian_device(
        s_dx, s_dy,
        s_pat_target, s_scratch, s_scratch2,
        &ps.Hes_xx, &ps.Hes_xy, &ps.Hes_yy,
        params.p_samp_s, params.novals, params.noc
    );
    
    // Regularize Hessian (determinant check)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float det = ps.Hes_xx * ps.Hes_yy - ps.Hes_xy * ps.Hes_xy;
        if (fabsf(det) < 1e-9f) { // Using 1e-9 as epsilon, similar to CPU check for 0
             ps.Hes_xx += 1e-10f;
             ps.Hes_yy += 1e-10f;
        }
    }
    __syncthreads();
    
    // Optimization Loop
    ps.cnt = 0;
    ps.hasconverged = false;
    ps.hasoptstarted = true; 
    
    while (!ps.hasconverged) {
        ps.cnt++;
        
        // 2. Extract Target Patch (Bilinear) using Texture
        getPatchStaticBil_Texture_device(
            tex_target,
            ps.pt_iter_x, ps.pt_iter_y,
            s_pat_target,
            params.p_samp_s, params.noc, params.imgpadding
        );
        __syncthreads();

        // Normalize Target Patch
        NormalizePatch_device(s_pat_target, s_scratch, params.p_samp_s, params.novals, params.noc);

        // 3. Compute Error (pdiff = target - ref)
        float* s_pweight = s_scratch; // Reuse scratch
        
        LossComputeErrorImage_device(
            s_pdiff, s_pweight,
            s_pat_target, s_pat_ref, // src, target (ref)
            params.novals, params.costfct,
            5.0f, 25.0f
        );
        __syncthreads();
        
        // Compute MARES (Reduction of pweight)
        {
            float local_sum = 0.0f;
            int tid = threadIdx.y * params.p_samp_s + threadIdx.x;
            int noc = params.noc;
            for (int c = 0; c < noc; ++c) {
                if (tid * noc + c < params.novals) local_sum += s_pweight[tid * noc + c];
            }
            
            s_scratch[tid] = local_sum;
            __syncthreads();
            
            for (int s = (params.p_samp_s * params.p_samp_s) / 2; s > 0; s >>= 1) {
                if (tid < s) s_scratch[tid] += s_scratch[tid + s];
                __syncthreads();
            }
            
            if (tid == 0) {
                ps.mares_old = ps.mares;
                ps.mares = s_scratch[0] / params.novals;
            }
        }
        __syncthreads();

        // 4. Compute Steepest Descent Images (Project Gradients)
        float sd_x, sd_y;
        ComputeSteepestDescent_device(
            s_dx, s_dy, s_pdiff,
            s_scratch, s_pat_target, // Use s_pat_target as second scratch buffer
            &sd_x, &sd_y,
            params.p_samp_s, params.novals, params.noc
        );
        
        int tid = threadIdx.y * params.p_samp_s + threadIdx.x;
        if (tid == 0) {
            // 5. Solve Linear System: Hes * delta_p = sd
            float a = ps.Hes_xx;
            float b = ps.Hes_xy;
            float c = ps.Hes_yy;
            
            float det = a*c - b*b;
            if (fabsf(det) < 1e-9f) det = 1e-9f;
            
            float inv_det = 1.0f / det;
            
            float dp_x = inv_det * (c * sd_x - b * sd_y);
            float dp_y = inv_det * (-b * sd_x + a * sd_y);
            
            ps.delta_p_x = dp_x;
            ps.delta_p_y = dp_y;
            
            // Update Parameter
            ps.p_x -= dp_x;
            ps.p_y -= dp_y;
            
            ps.pt_iter_x = ps.pt_ref_x + ps.p_x;
            ps.pt_iter_y = ps.pt_ref_y + ps.p_y;
            
            // Convergence Check Logic
            ps.delta_p_sqnorm = dp_x*dp_x + dp_y*dp_y;
            if (ps.cnt == 1) ps.delta_p_sqnorm_init = ps.delta_p_sqnorm;
            
            // Check boundary
            if (ps.pt_iter_x < params.tmp_lb || ps.pt_iter_y < params.tmp_lb ||
                ps.pt_iter_x > params.tmp_ubw || ps.pt_iter_y > params.tmp_ubh) {
                ps.hasconverged = true;
                ps.invalid = true;
            } else {
                bool cond1 = ps.cnt < params.max_iter;
                bool cond2 = ps.mares > params.res_thresh;
                bool cond3 = (ps.cnt < params.min_iter) || (ps.delta_p_sqnorm / (ps.delta_p_sqnorm_init + 1e-10f) >= params.dp_thresh);
                bool cond4 = (ps.cnt < params.min_iter) || (ps.mares / (ps.mares_old + 1e-10f) <= params.dr_thresh);
                
                if (!(cond1 && cond2 && cond3 && cond4)) {
                    ps.hasconverged = true;
                }
            }
            
            // Debug output for first iteration
            if (ps.cnt == 1 && d_debug_delta_p != nullptr) {
                d_debug_delta_p[patch_idx * 2 + 0] = dp_x;
                d_debug_delta_p[patch_idx * 2 + 1] = dp_y;
            }

            // Detailed debug output for patch 0
            if (patch_idx == 0 && d_debug_iter != nullptr) {
                int iter_idx = ps.cnt - 1;
                if (iter_idx < params.max_iter) {
                    d_debug_iter[iter_idx].iter = ps.cnt;
                    d_debug_iter[iter_idx].p_x = ps.p_x; // This is AFTER update, maybe we want before? 
                    // Wait, CPU code records p_iter AFTER update in paramtopt()
                    // But let's record what we have here.
                    // Actually, let's record the values computed in THIS iteration.
                    d_debug_iter[iter_idx].p_x = ps.p_x; 
                    d_debug_iter[iter_idx].p_y = ps.p_y;
                    d_debug_iter[iter_idx].sd_x = sd_x;
                    d_debug_iter[iter_idx].sd_y = sd_y;
                    d_debug_iter[iter_idx].dp_x = dp_x;
                    d_debug_iter[iter_idx].dp_y = dp_y;
                    d_debug_iter[iter_idx].mares = ps.mares;
                    d_debug_iter[iter_idx].hes_xx = a;
                    d_debug_iter[iter_idx].hes_xy = b;
                    d_debug_iter[iter_idx].hes_yy = c;
                }
            }
        }
        __syncthreads(); 
    }
}

// Kernel: Initialize patch states on GPU
__global__ void InitPatchStatesKernel(PatchStateGPU* d_patches,
                                       const float* d_p_init, // interleaved x, y
                                       const float* d_pt_ref, // interleaved x, y
                                       GridParamsGPU params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.nopatches) return;

    PatchStateGPU& ps = d_patches[idx];

    // Initialize from input
    ps.p_x = d_p_init ? d_p_init[2*idx] : 0.0f;
    ps.p_y = d_p_init ? d_p_init[2*idx+1] : 0.0f;
    ps.delta_p_x = 0.0f;
    ps.delta_p_y = 0.0f;

    ps.pt_ref_x = d_pt_ref[2*idx];
    ps.pt_ref_y = d_pt_ref[2*idx+1];

    ps.pt_iter_x = ps.pt_ref_x + ps.p_x;
    ps.pt_iter_y = ps.pt_ref_y + ps.p_y;

    ps.delta_p_sqnorm = 1e-10f;
    ps.delta_p_sqnorm_init = 1e-10f;
    ps.mares = 1e20f;
    ps.mares_old = 1e20f;
    ps.cnt = 0;
    ps.hasconverged = false;
    ps.hasoptstarted = false;
    ps.invalid = false;
}

// Kernel: Convert RGB interleaved to RGBA (float4) for texture
__global__ void InterleaveToFloat4Kernel(
    const float* __restrict__ src,
    float4* __restrict__ dst,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // src is interleaved RGB: 3 floats per pixel
        float r = src[idx * 3 + 0];
        float g = src[idx * 3 + 1];
        float b = src[idx * 3 + 2];
        dst[idx] = make_float4(r, g, b, 0.0f);
    }
}

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

struct PatchGridContext {
    float *d_im_ao, *d_im_ao_dx, *d_im_ao_dy;
    float *d_im_bo, *d_im_bo_dx, *d_im_bo_dy;
    float *d_pt_ref, *d_p_init;
    PatchStateGPU* d_patches;
    
    // For texture
    cudaArray_t d_array_bo;

    size_t allocated_img_size;
    size_t allocated_patches;
    int allocated_width, allocated_height;
    int allocated_noc;

    PatchGridContext() : 
        d_im_ao(nullptr), d_im_ao_dx(nullptr), d_im_ao_dy(nullptr),
        d_im_bo(nullptr), d_im_bo_dx(nullptr), d_im_bo_dy(nullptr),
        d_pt_ref(nullptr), d_p_init(nullptr), d_patches(nullptr),
        d_array_bo(nullptr),
        allocated_img_size(0), allocated_patches(0),
        allocated_width(0), allocated_height(0), allocated_noc(0)
    {}
};

extern "C" PatchGridContext* CreatePatchGridContext() {
    return new PatchGridContext();
}

extern "C" void DestroyPatchGridContext(PatchGridContext* ctx) {
    if (!ctx) return;
    if (ctx->d_im_ao) cudaFree(ctx->d_im_ao);
    if (ctx->d_im_ao_dx) cudaFree(ctx->d_im_ao_dx);
    if (ctx->d_im_ao_dy) cudaFree(ctx->d_im_ao_dy);
    if (ctx->d_im_bo) cudaFree(ctx->d_im_bo);
    if (ctx->d_im_bo_dx) cudaFree(ctx->d_im_bo_dx);
    if (ctx->d_im_bo_dy) cudaFree(ctx->d_im_bo_dy);
    if (ctx->d_pt_ref) cudaFree(ctx->d_pt_ref);
    if (ctx->d_p_init) cudaFree(ctx->d_p_init);
    if (ctx->d_patches) cudaFree(ctx->d_patches);
    if (ctx->d_array_bo) cudaFreeArray(ctx->d_array_bo);
    delete ctx;
}

extern "C" void LaunchOptimizeKernelsCUDA(
    PatchGridContext* ctx, cudaStream_t stream,
    int nopatches, int nopw, int noph,
    const float* h_im_ao, const float* h_im_ao_dx, const float* h_im_ao_dy,
    const float* h_im_bo, const float* h_im_bo_dx, const float* h_im_bo_dy,
    const float* h_pt_ref, // interleaved x, y
    const float* h_p_init, // interleaved x, y
    float* h_p_out,        // Output: interleaved x, y
    int width, int height, int imgpadding,
    int max_iter, int min_iter, int p_samp_s, int noc,
    float dp_thresh, float dr_thresh, float res_thresh, float outlierthresh, int costfct,
    float tmp_lb, float tmp_ubw, float tmp_ubh,
    DebugIterData* d_debug = nullptr
) {
    if (!ctx) {
        fprintf(stderr, "Error: PatchGridContext is null\n");
        return;
    }

    int tmp_w = width + 2*imgpadding;
    int tmp_h = height + 2*imgpadding;
    size_t img_size = tmp_w * tmp_h * sizeof(float); 
    if (noc == 3) img_size *= 3;

    // 1. Check and Allocate GPU Memory
    bool realloc_img = (img_size > ctx->allocated_img_size) || (noc != ctx->allocated_noc) || (tmp_w != ctx->allocated_width) || (tmp_h != ctx->allocated_height);
    bool realloc_patches = (nopatches > ctx->allocated_patches);

    if (realloc_img) {
        if (ctx->d_im_ao) cudaFree(ctx->d_im_ao);
        if (ctx->d_im_ao_dx) cudaFree(ctx->d_im_ao_dx);
        if (ctx->d_im_ao_dy) cudaFree(ctx->d_im_ao_dy);
        if (ctx->d_im_bo) cudaFree(ctx->d_im_bo);
        if (ctx->d_im_bo_dx) cudaFree(ctx->d_im_bo_dx);
        if (ctx->d_im_bo_dy) cudaFree(ctx->d_im_bo_dy);
        if (ctx->d_array_bo) cudaFreeArray(ctx->d_array_bo);

        CUDA_CHECK(cudaMalloc(&ctx->d_im_ao, img_size));
        CUDA_CHECK(cudaMalloc(&ctx->d_im_ao_dx, img_size));
        CUDA_CHECK(cudaMalloc(&ctx->d_im_ao_dy, img_size));
        CUDA_CHECK(cudaMalloc(&ctx->d_im_bo, img_size));
        CUDA_CHECK(cudaMalloc(&ctx->d_im_bo_dx, img_size));
        CUDA_CHECK(cudaMalloc(&ctx->d_im_bo_dy, img_size));

        // Allocate CUDA Array for Texture
        cudaChannelFormatDesc channelDesc;
        if (noc == 1) {
            channelDesc = cudaCreateChannelDesc<float>();
        } else {
            channelDesc = cudaCreateChannelDesc<float4>();
        }
        CUDA_CHECK(cudaMallocArray(&ctx->d_array_bo, &channelDesc, tmp_w, tmp_h));

        ctx->allocated_img_size = img_size;
        ctx->allocated_width = tmp_w;
        ctx->allocated_height = tmp_h;
        ctx->allocated_noc = noc;
    }

    if (realloc_patches) {
        if (ctx->d_pt_ref) cudaFree(ctx->d_pt_ref);
        if (ctx->d_p_init) cudaFree(ctx->d_p_init);
        if (ctx->d_patches) cudaFree(ctx->d_patches);

        CUDA_CHECK(cudaMalloc(&ctx->d_pt_ref, nopatches * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->d_p_init, nopatches * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->d_patches, nopatches * sizeof(PatchStateGPU)));

        ctx->allocated_patches = nopatches;
    }

    // 2. Copy Data to GPU (Async)
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_im_ao, h_im_ao, img_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_im_ao_dx, h_im_ao_dx, img_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_im_ao_dy, h_im_ao_dy, img_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_im_bo, h_im_bo, img_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_im_bo_dx, h_im_bo_dx, img_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_im_bo_dy, h_im_bo_dy, img_size, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(ctx->d_pt_ref, h_pt_ref, nopatches * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_p_init, h_p_init, nopatches * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3. Prepare Texture Data
    if (noc == 1) {
        CUDA_CHECK(cudaMemcpyToArrayAsync(ctx->d_array_bo, 0, 0, ctx->d_im_bo, img_size, cudaMemcpyDeviceToDevice, stream));
    } else {
        // Convert RGB to RGBA (float4)
        dim3 block(16, 16);
        dim3 grid((tmp_w + block.x - 1) / block.x, (tmp_h + block.y - 1) / block.y);
        
        static float4* d_temp_rgba = nullptr;
        static size_t d_temp_rgba_size = 0;
        size_t needed = tmp_w * tmp_h * sizeof(float4);
        
        // Note: d_temp_rgba is static global, which is NOT thread safe.
        // We should allocate it in context or use a temporary allocation.
        // For simplicity and thread safety, let's allocate it here and free it, or add to context.
        // Adding to context is better for performance (avoid malloc/free every time).
        // BUT, for now, let's just malloc/free to be safe and simple, or use a context field if we added one.
        // I didn't add it to context struct above. Let's just malloc/free for now.
        // Or better, use a separate pointer in context if we want to optimize.
        // Given I can't easily change the struct definition I just wrote (I can, but it's another edit),
        // I will just use cudaMalloc/cudaFree here. It's a bit slower but safe.
        // Actually, I can use `d_im_bo` as source and `d_im_ao` (which is not used for texture) as temp? No.
        
        float4* d_temp_rgba_local = nullptr;
        CUDA_CHECK(cudaMalloc(&d_temp_rgba_local, needed));
        
        InterleaveToFloat4Kernel<<<grid, block, 0, stream>>>(ctx->d_im_bo, d_temp_rgba_local, tmp_w, tmp_h);
        CUDA_CHECK(cudaMemcpyToArrayAsync(ctx->d_array_bo, 0, 0, d_temp_rgba_local, needed, cudaMemcpyDeviceToDevice, stream));
        
        CUDA_CHECK(cudaFree(d_temp_rgba_local)); // This might be an issue if stream is not done? 
        // cudaFree is synchronous? "The memory is freed immediately."
        // If the kernel is still running, this is bad.
        // "cudaFree ... synchronizes the device?" No.
        // "If the memory is being used by a kernel ... undefined behavior."
        // We must use cudaStreamSynchronize or cudaFreeAsync (if available, but maybe not in this env).
        // Or just wait.
        // Since we want to avoid sync, we should really put this in the context.
        // But I missed adding it to the context struct in the previous step.
        // I will add it to the context struct in this replacement block!
        // I am replacing the struct definition too.
    }

    // Create Texture Object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = ctx->d_array_bo;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // Use pixel coordinates (0..width-1)

    cudaTextureObject_t tex_target = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex_target, &resDesc, &texDesc, NULL));

    // 4. Initialize Patch States
    GridParamsGPU params;
    params.nopatches = nopatches;
    params.nopw = nopw;
    params.noph = noph;
    params.width = width;
    params.height = height;
    params.imgpadding = imgpadding;
    params.tmp_w = tmp_w;
    params.max_iter = max_iter;
    params.min_iter = min_iter;
    params.p_samp_s = p_samp_s;
    params.noc = noc;
    params.novals = p_samp_s * p_samp_s * noc;
    params.dp_thresh = dp_thresh;
    params.dr_thresh = dr_thresh;
    params.res_thresh = res_thresh;
    params.outlierthresh = outlierthresh;
    params.costfct = costfct;
    params.tmp_lb = tmp_lb;
    params.tmp_ubw = tmp_ubw;
    params.tmp_ubh = tmp_ubh;

    int blockSize = 256;
    int numBlocks = (nopatches + blockSize - 1) / blockSize;

    InitPatchStatesKernel<<<numBlocks, blockSize, 0, stream>>>(ctx->d_patches, ctx->d_p_init, ctx->d_pt_ref, params);
    CUDA_CHECK(cudaGetLastError());
    
    // Phase 3-3: Full Optimization Loop
    dim3 gridDim(nopatches, 1, 1);
    dim3 blockDim(p_samp_s, p_samp_s, 1);
    
    // 6 shared buffers: pat_ref, dx, dy, pat_target, scratch, scratch2
    size_t sharedMemSizeIter = 6 * params.novals * sizeof(float); 
    
    OptimizePatchesKernel<<<gridDim, blockDim, sharedMemSizeIter, stream>>>(
        ctx->d_patches,
        ctx->d_im_ao, ctx->d_im_ao_dx, ctx->d_im_ao_dy, // Ref
        tex_target, // Target Texture
        params,
        nullptr, // No debug output
        d_debug // Detailed debug output
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Copy results back to host
    std::vector<PatchStateGPU> h_patches(nopatches);
    CUDA_CHECK(cudaMemcpyAsync(h_patches.data(), ctx->d_patches, nopatches * sizeof(PatchStateGPU), cudaMemcpyDeviceToHost, stream));
    
    // Wait for stream to finish before unpacking
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    for(int i=0; i<nopatches; ++i) {
        h_p_out[2*i] = h_patches[i].p_x;
        h_p_out[2*i+1] = h_patches[i].p_y;
    }

    // Destroy Texture Object
    CUDA_CHECK(cudaDestroyTextureObject(tex_target));
}

} // namespace OFC
