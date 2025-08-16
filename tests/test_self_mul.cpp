#include "defs.h"
#include "Ec.h"
#include <cassert>

bool SelfTestMul()
{
    EcInt k; k.RndBits(128);
    EcPoint p_plain = Ec::MultiplyG(k);
    EcPoint p_glv = Ec::MultiplyG_GLV(k);
    return p_plain.IsEqual(p_glv);
}

int main()
{
    InitEc();
    for (int i = 0; i < 10; ++i) {
        assert(SelfTestMul());
    }
    return 0;
}
