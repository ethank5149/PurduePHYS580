#include <vector>
#include <cmath>

#include "initialize.h"
#include "norm.h"
#include "store.h"
#include "run.h"

using namespace std;

long double v_1(long double x){
    return pow(x,1);
}

long double v_2(long double x){
    return pow(x,2);
}

long double v_3(long double x){
    return pow(x,3);
}

long double v_4(long double x){
    return pow(x,4);
}

int main() {
    vector<long double> psi, x;
    long double (*V1)(long double) = &v_1;
    long double (*V2)(long double) = &v_2;
    long double (*V3)(long double) = &v_3;
    long double (*V4)(long double) = &v_4;

    initialize(psi,x);
    run(psi, x, V4, params.Ei, params.dE);
    norm(psi, x);
    store(psi, x);

    return 0;
}
