#include "Ec.h"
#include "RCGpuUtils.h"
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>

extern __device__ __constant__ u64 BETA[4];
extern __device__ __constant__ u64 BETA2[4];

__global__ void kernel_phi(u64* out1, u64* out2, u64* x) {
    MulModP(out1, x, BETA);
    MulModP(out2, x, BETA2);
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        printf("no gpu\n");
        return 1;
    }
    InitEc();
    EcInt beta2 = g_Beta;
    beta2.MulModP(g_Beta);
    cudaMemcpyToSymbol(BETA, g_Beta.data, 32);
    cudaMemcpyToSymbol(BETA2, beta2.data, 32);
    SetRndSeed(5);
    for (int i = 0; i < 4; i++) {
        EcInt k; k.RndBits(128);
        EcPoint P = Ec::MultiplyG(k);
        u64 *d_x, *d_o1, *d_o2;
        cudaMalloc(&d_x, 32);
        cudaMalloc(&d_o1, 32);
        cudaMalloc(&d_o2, 32);
        cudaMemcpy(d_x, P.x.data, 32, cudaMemcpyHostToDevice);
        kernel_phi<<<1,1>>>(d_o1, d_o2, d_x);
        cudaDeviceSynchronize();
        u64 o1[4], o2[4];
        cudaMemcpy(o1, d_o1, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(o2, d_o2, 32, cudaMemcpyDeviceToHost);
        cudaFree(d_x);
        cudaFree(d_o1);
        cudaFree(d_o2);
        EcInt cpu1 = P.x; cpu1.MulModP(g_Beta);
        EcInt cpu2 = P.x; cpu2.MulModP(beta2);
        if (memcmp(cpu1.data, o1, 32) != 0) {
            printf("beta mismatch at %d\n", i);
            return 1;
        }
        if (memcmp(cpu2.data, o2, 32) != 0) {
            printf("beta2 mismatch at %d\n", i);
            return 1;
        }
    }
    printf("test_phi_gpu: pass\n");
    return 0;
}
