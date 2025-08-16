#include "Ec.h"
#include "GpuKang.h"
#include <cuda_runtime.h>
#include <stdio.h>

extern void CallGpuKernelGen(TKparams Kparams);

static bool GpuCalcKG_local(EcPoint& out, const EcInt& k, int cuda_index) {
    TKparams params{};
    params.BlockCnt = 1;
    params.BlockSize = 1;
    params.GroupCnt = PNT_GROUP_CNT;
    params.KangCnt = PNT_GROUP_CNT;
    params.IsGenMode = true;

    if (cudaSetDevice(cuda_index) != cudaSuccess) return false;
    if (cudaMallocManaged((void**)&params.Kangs, params.KangCnt * 12 * sizeof(u64)) != cudaSuccess) return false;
    memset(params.Kangs, 0, params.KangCnt * 12 * sizeof(u64));
    params.Kangs[8] = k.data[0];
    params.Kangs[9] = k.data[1];
    params.Kangs[10] = 0;

    CallGpuKernelGen(params);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(params.Kangs);
        return false;
    }
    memcpy(out.x.data, &params.Kangs[0], 32);
    memcpy(out.y.data, &params.Kangs[4], 32);
    cudaFree(params.Kangs);
    return true;
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        printf("no gpu\n");
        return 1;
    }
    InitEc();
    SetRndSeed(1);
    for (int i = 0; i < 4; i++) {
        EcInt k; k.RndBits(128);
        EcPoint cpu = Ec::MultiplyG(k);
        EcPoint gpu;
        if (!GpuCalcKG_local(gpu, k, 0)) {
            printf("kernel error\n");
            return 1;
        }
        if (!cpu.IsEqual(gpu)) {
            printf("mismatch at %d\n", i);
            return 1;
        }
    }
    printf("test_mul_gpu: pass\n");
    return 0;
}
