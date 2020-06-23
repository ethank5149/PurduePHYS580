//
// Created by ethan on 6/22/2020.
//

#ifndef PROBLEM10_2_RUN_H

#include <vector>
#include "iterate.h"

using namespace std;

#define PROBLEM10_2_RUN_H

void run(vector<long double> &psi,vector<long double> &x,long double (*V)(long double), long double &E, long double &dE, long double eps = 1.0e-10){
    bool match, last_match;
    last_match = iterate(psi, x,V, E) >= 0.0;
    while(fabs(dE) > eps){
        match = iterate(psi, x, V,E) >= 0.0;
        if(match != last_match){
            dE = -0.5*dE;
        }
        last_match = match;
        E += dE;
    }
}

#endif //PROBLEM10_2_RUN_H
