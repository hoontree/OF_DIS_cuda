#ifndef REFINE_VARIATIONAL_CUDA_H
#define REFINE_VARIATIONAL_CUDA_H

#include <cuda_runtime.h>

extern "C" void LaunchVarRefCUDA(
    float* h_u, float* h_v, // Input/Output Flow (Host Pointers)
    const float* im1_c1, const float* im1_c2, const float* im1_c3,
    const float* im2_c1, const float* im2_c2, const float* im2_c3,
    int width, int height, int stride, int n_channels,
    int n_inner, int n_solver, 
    float alpha, float gamma, float delta, float omega
);

#endif // REFINE_VARIATIONAL_CUDA_H
