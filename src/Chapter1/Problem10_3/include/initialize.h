//
// Created by ethan on 6/22/2020.
//

#ifndef PROBLEM10_3_INITIALIZE_H

#include <vector>
#include <iostream>

using namespace std;

#define PROBLEM10_3_INITIALIZE_H

struct Parameters{
    int N{};
    long double Ei{};
    long double dE{};
    long double inf{};
    long double dx{};
}params;

void initialize(vector<long double> &psi, vector<long double> &x){
    cout << "Input: N Bound"<<endl;
    cin >> params.N >> params.inf;
    cout << endl;

    cout << "Input: E_i dE"<<endl;
    cin >> params.Ei>> params.dE;
    cout << endl;

    params.N = (int)params.N;
    if(params.N%2 != 0){
        params.N+=1;
    }

    x.resize(params.N);
    psi.resize(params.N);

    x[0] = -params.inf;
    params.dx = 2.0*params.inf/(params.N-1);

    for(unsigned int i=0;i<x.size()-1;i++){
        x[i+1] = x[i] + params.dx;
    }

    psi.front() = 0;
    psi.back() = 0;
    psi.at(1) = 0.0001*params.dx;
    psi.at(psi.size()-2) = 0.0001*params.dx;
}

#endif //PROBLEM10_3_INITIALIZE_H
