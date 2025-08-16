#include "Ec.h"
#include <cstdio>

int main() {
    InitEc();
    for (int i = 0; i < 1000; ++i) {
        EcInt k; k.RndBits(128);
        EcPoint p1 = Ec::MultiplyG(k);
        EcPoint p2 = Ec::MultiplyG_GLV(k);
        if (!p1.IsEqual(p2)) {
            printf("Mismatch at iteration %d\n", i);
            return 1;
        }
    }
    return 0;
}
