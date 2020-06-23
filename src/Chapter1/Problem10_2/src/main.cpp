//#include <iostream>
//#include <fstream>
//#include <string>
//#include <vector>
//#include <sstream>
//#include <limits>
//#include <iomanip>
//#include <cmath>
//#include <complex>
//
//using namespace std;
//
//const long double m = 1.0;
//const long double omega = 1.0;
//const long double eps = 1.0e-12;
//
//struct Parameters{
//    int N = 1000;
//    long double Ei = 0.0;
//    long double dE = 0.0;
//    long double inf = 10.0;
//    long double dx = 2.0/(long double)(N-1);
//}params;
//
//void norm(vector<long double> &psi, const vector<long double>& x){
//    const int N = psi.size();
//    const long double dx = x[1] - x[0];
//    long double f_i, f_ip1;
//    long double sum = 0.0;
//
//    for(unsigned int i=0;i<N-1;i++) {
//        f_i = (real(psi[i]) + imag(psi[i])) * (real(psi[i]) - imag(psi[i]));
//        f_ip1 = (real(psi[i + 1]) + imag(psi[i + 1])) * (real(psi[i + 1]) - imag(psi[i + 1]));
//        sum = sum + 0.5 * (f_i + f_ip1) * dx;
//    }
//
//    for(unsigned int j=0;j<N;j++) {
//        psi[j] = psi[j]/sqrt(sum);
//    }
//
//}
//
//long double obj_func(vector<long double> &psi, vector<long double> &x, long double E){
//    long double dx = x[1] - x[0];
//    long double vl, vr;
//    long double dleft, dright;
//    long int size = psi.size();
//    long int back_left = size/2-1;
//    long int front_right = size/2;
//
//    //Solve left
//    for(unsigned int j = 1; j<back_left;j++){
//        vl = 0.5*m*pow(omega*x[j],2);
//        psi[j+1] = 2.0 * (pow(dx,2) * (vl - E) + 1) * psi[j] - psi[j-1];
//    }
//    //Solve right
//    for(unsigned int j = size-2;j>front_right;j--){
//        vr = 0.5*m*pow(omega*x[j],2);
//        psi[j-1] = 2.0 * (pow(dx,2) * (vr - E) + 1) * psi[j] - psi[j+1];
//    }
//
//    //Scale depending on which is larger
//    if(psi.at(back_left)>psi.at(front_right)){
//        for(unsigned int j=0;j<front_right;j++) {
//            psi[j] = psi[j]*(psi.at(front_right)/psi.at(back_left));
//        }
//    }
//    else if(psi.at(back_left)<psi.at(front_right)){
//        for(unsigned int j=front_right;j<size;j++) {
//            psi[j] = psi[j]*(psi.at(back_left)/psi.at(front_right));
//        }
//    }
//
//    //Numerically calculate their derivatives
//    dleft = (psi.at(back_left)-psi.at(back_left-1))/(x.at(back_left)-x.at(back_left-1));
//    dright = (psi.at(front_right+1)-psi.at(front_right))/(x.at(front_right+1)-x.at(front_right));
//    if(dleft > dright){ return 1; }
//    else{ return -1; }
//}
//
//void initialize(vector<long double> &psi, vector<long double> &x){
//    cout << "Input: N Ei dE inf"<<endl;
//    cin >> params.N >> params.Ei>> params.dE >> params.inf;
//    cout << endl;
//
//    params.N = (int)params.N;
//    if(params.N%2 != 0){
//        params.N+=1;
//    }
//
//    x.resize(params.N);
//    psi.resize(params.N);
//
//    x[0] = -params.inf;
//    params.dx = 2.0*params.inf/(params.N-1);
//
//    for(unsigned int i=0;i<x.size()-1;i++){
//        x[i+1] = x[i] + params.dx;
//    }
//
//    psi.front() = 0;
//    psi.back() = 0;
//    psi.at(1) = 0.0001*params.dx;
//    psi.at(psi.size()-2) = 0.0001*params.dx;
//
//}
//
//int main() {
//    vector<long double> psi, x;
//    long double prod;
//    bool match, last_match;
//
//    initialize(psi,x);
//    long double E = params.Ei;
//    long double dE = params.dE;
//
//    prod = obj_func(psi, x, E);
//    last_match = prod >= 0.0;
//    cout << setprecision(numeric_limits<long double>::digits10 + 1);
//    while(fabs(dE) > eps){
//        cout << E << endl;
//        prod = obj_func(psi, x, E);
//        match = prod >= 0.0;
//        if(match != last_match){
//            dE = -0.5*dE;
//        }
//        last_match = match;
//        E += dE;
//    }
//    norm(psi, x);
//
//    ofstream file("../data/output.dat");
//    file << setprecision(numeric_limits<long double>::digits10 + 1);
//    file << R"(# x \psi)" << endl;
//    for(unsigned int i=0;i<psi.size();i++){
//        file << x[i] << " " << psi[i] << endl;
//    }
//
//    cout << setprecision(numeric_limits<long double>::digits10 + 1);
//    cout << "dE = " << dE << endl;
//    cout << "E = " << E << endl;
//    return 0;
//}

#include <vector>
#include <cmath>

#include "initialize.h"
#include "norm.h"
#include "store.h"
#include "run.h"

using namespace std;

const long double m = 1.0;
const long double omega = 1.0;

long double v_qho(long double x){
    return 0.5*m*pow(omega*x,2);
}

int main() {
    vector<long double> psi, x;
    long double (*V)(long double) = &v_qho;

    initialize(psi,x);
    run(psi, x, V, params.Ei, params.dE);
    norm(psi, x);
    store(psi, x);

    return 0;
}
