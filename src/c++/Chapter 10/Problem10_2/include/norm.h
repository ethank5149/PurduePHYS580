//
// Created by ethan on 6/22/2020.
//

#ifndef PROBLEM10_2_NORM_H

#include <vector>
#include <cmath>
#include <complex>

using namespace std;

#define PROBLEM10_2_NORM_H

void norm(vector<long double> &psi, const vector<long double>& x){
    const int N = psi.size();
    const long double dx = x[1] - x[0];
    long double f_i, f_ip1;
    long double sum = 0.0;

    for(unsigned int i=0;i<N-1;i++) {
        f_i = (real(psi[i]) + imag(psi[i])) * (real(psi[i]) - imag(psi[i]));
        f_ip1 = (real(psi[i + 1]) + imag(psi[i + 1])) * (real(psi[i + 1]) - imag(psi[i + 1]));
        sum = sum + 0.5 * (f_i + f_ip1) * dx;
    }

    for(unsigned int j=0;j<N;j++) {
        psi[j] = psi[j]/sqrt(sum);
    }

}


#endif //PROBLEM10_2_NORM_H
