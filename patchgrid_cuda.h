#ifndef PATCHGRID_CUDA_H
#define PATCHGRID_CUDA_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context structure (defined in .cu)
typedef struct PatchGridContext PatchGridContext;

// Create and destroy context
PatchGridContext* CreatePatchGridContext();
void DestroyPatchGridContext(PatchGridContext* ctx);

// Debug structure for iteration data
typedef struct {
    int iter;
    float p_x, p_y;
    float sd_x, sd_y;
    float dp_x, dp_y;
    float mares;
    float hes_xx, hes_xy, hes_yy;
} DebugIterData;

// CUDA kernel launcher (implemented in patchgrid_cuda.cu)
// Called from patchgrid.cpp with extern "C" linkage
// Parameters: number of patches, grid width, grid height
void LaunchOptimizeKernelsCUDA(
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
);

#ifdef __cplusplus
}
#endif



#endif // PATCHGRID_CUDA_H
