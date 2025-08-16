#include "defs.h"
#include "Ec.h"
#include <cassert>

int main()
{
    InitEc();

    // baseline: compare MultiplyG and MultiplyG_GLV on a non-zero scalar
    {
        EcInt k; k.Set(123456789);
        EcPoint p1 = Ec::MultiplyG(k);
        EcPoint p2 = Ec::MultiplyG_GLV(k);
        assert(p1.x.IsEqual(p2.x));
        assert(p1.y.IsEqual(p2.y));
    }

    // explicit check for zero scalar
    {
        EcInt zero; zero.SetZero();
        EcPoint pg = Ec::MultiplyG(zero);
        EcPoint pglv = Ec::MultiplyG_GLV(zero);
        assert(pg.x.IsEqual(pglv.x));
        assert(pg.y.IsEqual(pglv.y));
    }

    return 0;
}
