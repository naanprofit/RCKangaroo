#include "defs.h"
#include "Ec.h"
#include <cassert>
#include <cstdio>

int main()
{
    InitEc();

    EcInt key; key.Set(12345);
    EcInt w; w.Set(0);
    EcInt t = key; t.Add(w);

    // Scenario with k = 1 (lambda)
    {
        EcInt t1 = t, w1 = w;
        t1.MulLambdaN();
        w1.MulLambdaN();
        EcInt res = t1; res.Sub(w1);
        res.MulLambda2N();
        assert(res.IsEqual(key));
    }

    // Scenario with k = 2 (lambda^2)
    {
        EcInt t2 = t, w2 = w;
        t2.MulLambda2N();
        w2.MulLambda2N();
        EcInt res = t2; res.Sub(w2);
        res.MulLambdaN();
        assert(res.IsEqual(key));
    }

    printf("test_lambda: pass\n");
    return 0;
}
