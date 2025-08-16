#include "Ec.h"
#include "GpuKang.h"
#include <cuda_runtime.h>

bool GpuCalcKG(EcPoint& out, const EcInt& k, int cuda_index);

int main() {
    if (cudaSetDevice(0) != cudaSuccess) return 1;
    SetRndSeed(1);
    for (int i = 0; i < 4; i++) {
        EcInt k; k.RndBits(128);
        EcPoint cpu = Ec::MultiplyG(k);
        EcPoint gpu;
        if (!GpuCalcKG(gpu, k, 0)) return 1;
        if (!cpu.IsEqual(gpu)) return 1;
    }
    return 0;
}
