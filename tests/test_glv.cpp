#include "Ec.h"
#include <cassert>
int main(){
    Ec ec; InitEc();
    SetRndSeed(0);
    for(int i=0;i<100;i++){
        EcInt k; k.RndBits(128);
        EcPoint a = ec.MultiplyG(k);
        EcPoint b = ec.MultiplyG_GLV(k);
        assert(a.IsEqual(b));
    }
    return 0;
}
