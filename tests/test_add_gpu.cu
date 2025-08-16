#include "Ec.h"
#include "RCGpuUtils.h"
#include <cuda_runtime.h>
#include <string.h>

__device__ __forceinline__ void AddPoints(u64* res_x, u64* res_y, u64* p1x, u64* p1y, u64* p2x, u64* p2y) {
    __align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
    __align__(16) u64 inverse[5];
    SubModP(inverse, p2x, p1x);
    InvModP((u32*)inverse);
    SubModP(tmp, p2y, p1y);
    MulModP(lambda, tmp, inverse);
    MulModP(lambda2, lambda, lambda);
    SubModP(tmp, lambda2, p1x);
    SubModP(res_x, tmp, p2x);
    SubModP(tmp, p2x, res_x);
    MulModP(tmp2, tmp, lambda);
    SubModP(res_y, tmp2, p2y);
}

__global__ void kernel_add(u64* out, u64* p1, u64* p2) {
    AddPoints(out, out + 4, p1, p1 + 4, p2, p2 + 4);
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) return 1;
    SetRndSeed(3);
    for (int i = 0; i < 4; i++) {
        EcInt k1, k2; k1.RndBits(128); k2.RndBits(128);
        EcPoint p1 = Ec::MultiplyG(k1);
        EcPoint p2 = Ec::MultiplyG(k2);
        EcPoint cpu = Ec::AddPoints(p1, p2);
        u64 *d_p1, *d_p2, *d_out;
        cudaMalloc(&d_p1, 64);
        cudaMalloc(&d_p2, 64);
        cudaMalloc(&d_out, 64);
        cudaMemcpy(d_p1, p1.x.data, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_p1 + 4, p1.y.data, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_p2, p2.x.data, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_p2 + 4, p2.y.data, 32, cudaMemcpyHostToDevice);
        kernel_add<<<1,1>>>(d_out, d_p1, d_p2);
        cudaDeviceSynchronize();
        u64 out[8];
        cudaMemcpy(out, d_out, 64, cudaMemcpyDeviceToHost);
        cudaFree(d_p1);
        cudaFree(d_p2);
        cudaFree(d_out);
        EcPoint gpu;
        memcpy(gpu.x.data, out, 32);
        memcpy(gpu.y.data, out + 4, 32);
        if (!gpu.IsEqual(cpu)) return 1;
    }
    return 0;
}
