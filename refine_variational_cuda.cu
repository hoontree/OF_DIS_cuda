#include "refine_variational_cuda.h"
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------

__global__ void InterleaveKernel(
    const float* c1, const float* c2, const float* c3,
    float4* dst,
    int width, int height, int stride, int noc
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * stride + x;
    int out_idx = y * width + x;
    
    float v1 = c1[idx];
    float v2 = (noc >= 2 && c2) ? c2[idx] : 0.0f;
    float v3 = (noc >= 3 && c3) ? c3[idx] : 0.0f;
    
    dst[out_idx] = make_float4(v1, v2, v3, 0.0f);
}

// ... WarpDeriv1, WarpDeriv3, Smoothness, Data, SubLap, SOR, UpdateFlow ...
// (I will retain previous implementations, just adding the changes for Tex and Interleave safety)

// Warp & Compute Derivatives (Single Channel)
__global__ void WarpDerivKernel1(
    const float* d_im1, 
    cudaTextureObject_t tex_im2,
    const float* d_u, const float* d_v,
    float* d_Ix, float* d_Iy, float* d_Iz,
    int width, int height, int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * stride + x;
    
    float u = x + d_u[idx];
    float v = y + d_v[idx];
    
    // Warped I2: Texture is float4, we read .x
    // Note: If using float4 texture for single channel, read .x
    float i2 = tex2D<float4>(tex_im2, u + 0.5f, v + 0.5f).x;
    float i1 = d_im1[idx];
    
    if (d_Iz) d_Iz[idx] = i2 - i1;
    
    float i2_x = (tex2D<float4>(tex_im2, u + 1.5f, v + 0.5f).x - tex2D<float4>(tex_im2, u - 0.5f, v + 0.5f).x) * 0.5f;
    float i2_y = (tex2D<float4>(tex_im2, u + 0.5f, v + 1.5f).x - tex2D<float4>(tex_im2, u + 0.5f, v - 0.5f).x) * 0.5f;
    
    float i1_l = (x > 0) ? d_im1[idx - 1] : i1;
    float i1_r = (x < width - 1) ? d_im1[idx + 1] : i1;
    float i1_x = (i1_r - i1_l) * 0.5f;
    
    float i1_u = (y > 0) ? d_im1[idx - stride] : i1;
    float i1_d = (y < height - 1) ? d_im1[idx + stride] : i1;
    float i1_y = (i1_d - i1_u) * 0.5f;
    
    if (d_Ix) d_Ix[idx] = 0.5f * (i1_x + i2_x);
    if (d_Iy) d_Iy[idx] = 0.5f * (i1_y + i2_y);
}

__global__ void WarpDerivKernel3(
    const float* c1, const float* c2, const float* c3, 
    cudaTextureObject_t tex_im2, 
    const float* d_u, const float* d_v,
    float* d_Ix, float* d_Iy, float* d_Iz, 
    int width, int height, int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * stride + x;
    int plane_off = stride * height;
    
    float u = x + d_u[idx];
    float v = y + d_v[idx];
    
    float4 i2_val = tex2D<float4>(tex_im2, u + 0.5f, v + 0.5f);
    
    // Channel 0
    float i1_0 = c1[idx];
    d_Iz[idx] = i2_val.x - i1_0;
    
    float i2_x_0 = (tex2D<float4>(tex_im2, u + 1.5f, v + 0.5f).x - tex2D<float4>(tex_im2, u - 0.5f, v + 0.5f).x) * 0.5f;
    float i1_r = (x < width - 1) ? c1[idx+1] : i1_0;
    float i1_l = (x > 0) ? c1[idx-1] : i1_0;
    float i1_x_0 = (i1_r - i1_l) * 0.5f;
    d_Ix[idx] = 0.5f * (i1_x_0 + i2_x_0);
    
    float i2_y_0 = (tex2D<float4>(tex_im2, u + 0.5f, v + 1.5f).x - tex2D<float4>(tex_im2, u + 0.5f, v - 0.5f).x) * 0.5f;
    float i1_d = (y < height - 1) ? c1[idx+stride] : i1_0;
    float i1_u_val = (y > 0) ? c1[idx-stride] : i1_0;
    float i1_y_0 = (i1_d - i1_u_val) * 0.5f;
    d_Iy[idx] = 0.5f * (i1_y_0 + i2_y_0);
    
    // Channel 1
    float i1_1 = c2[idx];
    d_Iz[idx + plane_off] = i2_val.y - i1_1;
    
    float i2_x_1 = (tex2D<float4>(tex_im2, u + 1.5f, v + 0.5f).y - tex2D<float4>(tex_im2, u - 0.5f, v + 0.5f).y) * 0.5f;
    i1_r = (x < width - 1) ? c2[idx+1] : i1_1;
    i1_l = (x > 0) ? c2[idx-1] : i1_1;
    float i1_x_1 = (i1_r - i1_l) * 0.5f;
    d_Ix[idx + plane_off] = 0.5f * (i1_x_1 + i2_x_1);
    
    float i2_y_1 = (tex2D<float4>(tex_im2, u + 0.5f, v + 1.5f).y - tex2D<float4>(tex_im2, u + 0.5f, v - 0.5f).y) * 0.5f;
    i1_d = (y < height - 1) ? c2[idx+stride] : i1_1;
    i1_u_val = (y > 0) ? c2[idx-stride] : i1_1;
    float i1_y_1 = (i1_d - i1_u_val) * 0.5f;
    d_Iy[idx + plane_off] = 0.5f * (i1_y_1 + i2_y_1);
    
    // Channel 2
    float i1_2 = c3[idx];
    d_Iz[idx + 2*plane_off] = i2_val.z - i1_2;
    
    float i2_x_2 = (tex2D<float4>(tex_im2, u + 1.5f, v + 0.5f).z - tex2D<float4>(tex_im2, u - 0.5f, v + 0.5f).z) * 0.5f;
    i1_r = (x < width - 1) ? c3[idx+1] : i1_2;
    i1_l = (x > 0) ? c3[idx-1] : i1_2;
    float i1_x_2 = (i1_r - i1_l) * 0.5f;
    d_Ix[idx + 2*plane_off] = 0.5f * (i1_x_2 + i2_x_2);
    
    float i2_y_2 = (tex2D<float4>(tex_im2, u + 0.5f, v + 1.5f).z - tex2D<float4>(tex_im2, u + 0.5f, v - 0.5f).z) * 0.5f;
    i1_d = (y < height - 1) ? c3[idx+stride] : i1_2;
    i1_u_val = (y > 0) ? c3[idx-stride] : i1_2;
    float i1_y_2 = (i1_d - i1_u_val) * 0.5f;
    d_Iy[idx + 2*plane_off] = 0.5f * (i1_y_2 + i2_y_2);
}

__global__ void ComputeSmoothnessPsiKernel(
    const float* d_u, const float* d_v, 
    float* d_psi_h, float* d_psi_v,
    float alpha, int width, int height, int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * stride + x;

    float u_val = d_u[idx];
    float v_val = d_v[idx];
    
    float ux = (x < width - 1) ? (d_u[idx + 1] - u_val) : 0.0f;
    float vx = (x < width - 1) ? (d_v[idx + 1] - v_val) : 0.0f;
    
    float eps_sq = 1e-6f;
    d_psi_h[idx] = 1.0f / sqrtf(ux*ux + vx*vx + eps_sq);

    float uy = (y < height - 1) ? (d_u[idx + stride] - u_val) : 0.0f;
    float vy = (y < height - 1) ? (d_v[idx + stride] - v_val) : 0.0f;
    d_psi_v[idx] = 1.0f / sqrtf(uy*uy + vy*vy + eps_sq);
}

__global__ void ComputeDataTermThisKernel(
    const float* d_Ix, const float* d_Iy, const float* d_Iz,
    const float* d_du, const float* d_dv, 
    float* d_a11, float* d_a12, float* d_a22,
    float* d_b1, float* d_b2,
    int width, int height, int stride, int noc
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * stride + x;
    int plane_off = stride * height;
    
    float sum_a11 = 0.0f, sum_a12 = 0.0f, sum_a22 = 0.0f;
    float sum_b1 = 0.0f, sum_b2 = 0.0f;
    float du = d_du[idx];
    float dv = d_dv[idx];
    
    float eps_sq = 1e-6f;

    for(int k=0; k<noc; ++k) {
        int off = k * plane_off;
        float ix = d_Ix[idx + off];
        float iy = d_Iy[idx + off];
        float iz = d_Iz[idx + off];
        
        float r = iz + ix*du + iy*dv;
        float psi = 1.0f / sqrtf(r*r + eps_sq);
        
        sum_a11 += psi * ix * ix;
        sum_a12 += psi * ix * iy;
        sum_a22 += psi * iy * iy;
        
        sum_b1 += -psi * iz * ix;
        sum_b2 += -psi * iz * iy;
    }
    
    d_a11[idx] = sum_a11;
    d_a12[idx] = sum_a12;
    d_a22[idx] = sum_a22;
    d_b1[idx] = sum_b1;
    d_b2[idx] = sum_b2;
}

__global__ void SubLaplacianKernel(
    float* d_b1, float* d_b2,
    const float* d_u, const float* d_v,
    const float* d_psi_h, const float* d_psi_v,
    float alpha,
    int width, int height, int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * stride + x;
    
    float u_c = d_u[idx];
    float v_c = d_v[idx];
    
    // East
    float w_E = (x < width - 1) ? d_psi_h[idx] : 0.0f;
    float u_E = (x < width - 1) ? d_u[idx + 1] : 0.0f;
    float v_E = (x < width - 1) ? d_v[idx + 1] : 0.0f;
    
    // West
    float w_W = (x > 0) ? d_psi_h[idx - 1] : 0.0f;
    float u_W = (x > 0) ? d_u[idx - 1] : 0.0f;
    float v_W = (x > 0) ? d_v[idx - 1] : 0.0f;
    
    float L_u = 0.0f;
    float L_v = 0.0f;
    
    L_u += w_E * (u_E - u_c);
    L_u += w_W * (u_W - u_c);
    L_v += w_E * (v_E - v_c);
    L_v += w_W * (v_W - v_c);
    
    // South
    float w_S = (y < height - 1) ? d_psi_v[idx] : 0.0f;
    float u_S = (y < height - 1) ? d_u[idx + stride] : 0.0f;
    float v_S = (y < height - 1) ? d_v[idx + stride] : 0.0f;
    L_u += w_S * (u_S - u_c);
    L_v += w_S * (v_S - v_c);
    
    // North
    float w_N = (y > 0) ? d_psi_v[idx - stride] : 0.0f;
    float u_N = (y > 0) ? d_u[idx - stride] : 0.0f;
    float v_N = (y > 0) ? d_v[idx - stride] : 0.0f;
    L_u += w_N * (u_N - u_c);
    L_v += w_N * (v_N - v_c);
    
    d_b1[idx] += alpha * L_u;
    d_b2[idx] += alpha * L_v;
}

__global__ void SORKernel(
    float* d_du, float* d_dv,
    const float* d_a11, const float* d_a12, const float* d_a22,
    const float* d_b1, const float* d_b2,
    const float* d_psi_h, const float* d_psi_v,
    float alpha, float omega,
    int width, int height, int stride,
    int parity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if ((x + y) % 2 != parity) return;
    
    int idx = y * stride + x;
    
    float w_E = (x < width - 1) ? d_psi_h[idx] : 0.0f;
    float w_W = (x > 0)         ? d_psi_h[idx - 1] : 0.0f;
    float w_S = (y < height - 1) ? d_psi_v[idx] : 0.0f;
    float w_N = (y > 0)         ? d_psi_v[idx - stride] : 0.0f;
    
    float sum_w = w_E + w_W + w_S + w_N;
    
    float val_u_E = (x < width - 1) ? d_du[idx + 1] : 0.0f;
    float val_v_E = (x < width - 1) ? d_dv[idx + 1] : 0.0f;
    float val_u_W = (x > 0)         ? d_du[idx - 1] : 0.0f;
    float val_v_W = (x > 0)         ? d_dv[idx - 1] : 0.0f;
    float val_u_S = (y < height - 1) ? d_du[idx + stride] : 0.0f;
    float val_v_S = (y < height - 1) ? d_dv[idx + stride] : 0.0f;
    float val_u_N = (y > 0)         ? d_du[idx - stride] : 0.0f;
    float val_v_N = (y > 0)         ? d_dv[idx - stride] : 0.0f;
    
    float sigma_u = w_E*val_u_E + w_W*val_u_W + w_S*val_u_S + w_N*val_u_N;
    float sigma_v = w_E*val_v_E + w_W*val_v_W + w_S*val_v_S + w_N*val_v_N;
    
    float A_smooth = alpha * sum_w;
    
    float A11 = d_a11[idx] + A_smooth;
    float A12 = d_a12[idx];
    float A22 = d_a22[idx] + A_smooth;
    
    float B1 = d_b1[idx] + alpha * sigma_u;
    float B2 = d_b2[idx] + alpha * sigma_v;
    
    float det = A11 * A22 - A12 * A12;
    float new_du = d_du[idx];
    float new_dv = d_dv[idx];
    
    if (fabsf(det) > 1e-9f) {
        float inv_det = 1.0f / det;
        new_du = inv_det * (A22 * B1 - A12 * B2);
        new_dv = inv_det * (-A12 * B1 + A11 * B2);
    }
    
    d_du[idx] = (1.0f - omega) * d_du[idx] + omega * new_du;
    d_dv[idx] = (1.0f - omega) * d_dv[idx] + omega * new_dv;
}

__global__ void UpdateFlowKernel(
    float* d_u, float* d_v,
    const float* d_du, const float* d_dv,
    int width, int height, int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * stride + x;
    
    d_u[idx] += d_du[idx];
    d_v[idx] += d_dv[idx];
}

extern "C" void LaunchVarRefCUDA(
    float* h_u, float* h_v, 
    const float* im1_c1, const float* im1_c2, const float* im1_c3,
    const float* im2_c1, const float* im2_c2, const float* im2_c3,
    int width, int height, int stride, int noc,
    int n_inner, int n_solver, 
    float alpha, float gamma, float delta, float omega
) {
    // printf("DEBUG: LaunchVarRefCUDA noc=%d width=%d height=%d stride=%d\n", noc, width, height, stride);
    
    float *d_u, *d_v, *d_du, *d_dv;
    float *d_im1_c1, *d_im1_c2, *d_im1_c3;
    float *d_psi_h, *d_psi_v;
    float *d_Ix, *d_Iy, *d_Iz;
    float *d_a11, *d_a12, *d_a22, *d_b1, *d_b2;
    float4 *d_im2_interleaved_linear; // Temporary linear buffer
    cudaArray_t d_im2_array; // Texture Array
    
    size_t size_plane = stride * height * sizeof(float);
    
    // Allocations
    CUDA_CHECK(cudaMalloc(&d_u, size_plane));
    CUDA_CHECK(cudaMalloc(&d_v, size_plane));
    CUDA_CHECK(cudaMalloc(&d_du, size_plane));
    CUDA_CHECK(cudaMalloc(&d_dv, size_plane));
    
    CUDA_CHECK(cudaMalloc(&d_psi_h, size_plane));
    CUDA_CHECK(cudaMalloc(&d_psi_v, size_plane));
    
    CUDA_CHECK(cudaMalloc(&d_Ix, size_plane * noc));
    CUDA_CHECK(cudaMalloc(&d_Iy, size_plane * noc));
    CUDA_CHECK(cudaMalloc(&d_Iz, size_plane * noc));
    
    CUDA_CHECK(cudaMalloc(&d_a11, size_plane));
    CUDA_CHECK(cudaMalloc(&d_a12, size_plane));
    CUDA_CHECK(cudaMalloc(&d_a22, size_plane));
    CUDA_CHECK(cudaMalloc(&d_b1, size_plane));
    CUDA_CHECK(cudaMalloc(&d_b2, size_plane));
    
    // Copy Images
    CUDA_CHECK(cudaMalloc(&d_im1_c1, size_plane));
    CUDA_CHECK(cudaMemcpy(d_im1_c1, im1_c1, size_plane, cudaMemcpyHostToDevice));
    if (noc == 3) {
        CUDA_CHECK(cudaMalloc(&d_im1_c2, size_plane));
        CUDA_CHECK(cudaMalloc(&d_im1_c3, size_plane));
        CUDA_CHECK(cudaMemcpy(d_im1_c2, im1_c2, size_plane, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_im1_c3, im1_c3, size_plane, cudaMemcpyHostToDevice));
    }
    
    // Prepare Texture for im2
    float *d_im2_src[3] = {nullptr, nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_im2_src[0], size_plane));
    CUDA_CHECK(cudaMemcpy(d_im2_src[0], im2_c1, size_plane, cudaMemcpyHostToDevice));
    if (noc == 3) {
        CUDA_CHECK(cudaMalloc(&d_im2_src[1], size_plane));
        CUDA_CHECK(cudaMalloc(&d_im2_src[2], size_plane));
        CUDA_CHECK(cudaMemcpy(d_im2_src[1], im2_c2, size_plane, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_im2_src[2], im2_c3, size_plane, cudaMemcpyHostToDevice));
    }
    
    // Interleave im2 for Texture (to CUDA Array)
    // 1. Interleave to Linear
    size_t size_float4 = width * height * sizeof(float4);
    CUDA_CHECK(cudaMalloc(&d_im2_interleaved_linear, size_float4));
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    InterleaveKernel<<<grid, block>>>(d_im2_src[0], d_im2_src[1], d_im2_src[2], d_im2_interleaved_linear, width, height, stride, noc);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 2. Copy Linear to Array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&d_im2_array, &channelDesc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(d_im2_array, 0, 0, d_im2_interleaved_linear, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice));
    
    // Free intermediate
    CUDA_CHECK(cudaFree(d_im2_interleaved_linear));
    
    // Create Texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_im2_array;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex_im2 = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex_im2, &resDesc, &texDesc, NULL));
    
    // Copy Flow
    CUDA_CHECK(cudaMemcpy(d_u, h_u, size_plane, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, size_plane, cudaMemcpyHostToDevice));
    
    // Loop
    for (int i = 0; i < n_inner; ++i) {
        CUDA_CHECK(cudaMemset(d_du, 0, size_plane));
        CUDA_CHECK(cudaMemset(d_dv, 0, size_plane));
        
        // 1. Warp & Deriv
        if (noc == 1) {
            WarpDerivKernel1<<<grid, block>>>(d_im1_c1, tex_im2, d_u, d_v, d_Ix, d_Iy, d_Iz, width, height, stride);
        } else {
            WarpDerivKernel3<<<grid, block>>>(d_im1_c1, d_im1_c2, d_im1_c3, tex_im2, d_u, d_v, d_Ix, d_Iy, d_Iz, width, height, stride);
        }
        
        // 2. Smoothness Map
        ComputeSmoothnessPsiKernel<<<grid, block>>>(d_u, d_v, d_psi_h, d_psi_v, alpha * 0.25f, width, height, stride);
        
        // 3. Data Term
        ComputeDataTermThisKernel<<<grid, block>>>(d_Ix, d_Iy, d_Iz, d_du, d_dv, d_a11, d_a12, d_a22, d_b1, d_b2, width, height, stride, noc);
        
        // 4. Sub Laplacian
        SubLaplacianKernel<<<grid, block>>>(d_b1, d_b2, d_u, d_v, d_psi_h, d_psi_v, alpha * 0.25f, width, height, stride); 
        
        // 5. Solver
        for (int k = 0; k < n_solver; ++k) {
            SORKernel<<<grid, block>>>(d_du, d_dv, d_a11, d_a12, d_a22, d_b1, d_b2, d_psi_h, d_psi_v, alpha * 0.25f, omega, width, height, stride, 0); 
            SORKernel<<<grid, block>>>(d_du, d_dv, d_a11, d_a12, d_a22, d_b1, d_b2, d_psi_h, d_psi_v, alpha * 0.25f, omega, width, height, stride, 1); 
        }
        
        // 6. Update Flow
        UpdateFlowKernel<<<grid, block>>>(d_u, d_v, d_du, d_dv, width, height, stride);
    }
    
    // Copy Back
    CUDA_CHECK(cudaMemcpy(h_u, d_u, size_plane, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, d_v, size_plane, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaDestroyTextureObject(tex_im2));
    CUDA_CHECK(cudaFreeArray(d_im2_array));
    
    CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_du)); CUDA_CHECK(cudaFree(d_dv));
    CUDA_CHECK(cudaFree(d_psi_h)); CUDA_CHECK(cudaFree(d_psi_v));
    CUDA_CHECK(cudaFree(d_Ix)); CUDA_CHECK(cudaFree(d_Iy)); CUDA_CHECK(cudaFree(d_Iz));
    CUDA_CHECK(cudaFree(d_a11)); CUDA_CHECK(cudaFree(d_a12)); CUDA_CHECK(cudaFree(d_a22));
    CUDA_CHECK(cudaFree(d_b1)); CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_im1_c1)); 
    if (noc==3) { CUDA_CHECK(cudaFree(d_im1_c2)); CUDA_CHECK(cudaFree(d_im1_c3)); }
    CUDA_CHECK(cudaFree(d_im2_src[0])); 
    if (noc==3) { CUDA_CHECK(cudaFree(d_im2_src[1])); CUDA_CHECK(cudaFree(d_im2_src[2])); }
}
