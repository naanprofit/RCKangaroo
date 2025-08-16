#include "defs.h"
#include "Ec.h"
#include <cassert>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdio>

int main() {
    InitEc();
    using boost::multiprecision::cpp_int;

    auto to_cpp_int = [](const EcInt& x) {
        cpp_int r = 0;
        for (int i = 3; i >= 0; --i) {
            r <<= 64; r += x.data[i];
        }
        return r;
    };

    cpp_int beta = to_cpp_int(g_Beta);
    cpp_int P("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    cpp_int beta3 = (beta * beta % P) * beta % P;
    assert(beta3 == 1);
    assert(beta != 1);
    printf("test_beta: pass\n");
    return 0;
}
