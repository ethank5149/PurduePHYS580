#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <limits>
#include <iomanip>
#include <cmath>
#include <complex>

using namespace std;

const long double Vinf = 100.0;
const long double L = 10.0;
const long double l = 0.0;
const long double V0 = 2.0;
const long double zero = 1.0e-10;
const long double eps = 1.0e-8;

struct Parameters{
    int N = 1000;
    int N_E = 1000;

    long double Ei = 0.0;
    long double Ef = 10.0;
    long double inf = 10.0;
    long double xi = -inf;
    long double xf = inf;
    long double dx = (xf-xi)/(long double)(N-1);
    long double dE = (Ef-Ei)/(long double)(N_E-1);
    string method = "qho";
}params;

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

void solve_qho(vector<long double> &psi, vector<long double> &x, long double E){
    long double dx = x[1] - x[0];
    long double v;
    for(unsigned int j = 1; j<x.size()-1;j++){
        v = 0.5 * pow(x[j],2);
        psi[j+1] = 2.0 * (pow(dx,2) * (v - E) + 1) * psi[j] - psi[j-1];
    }
    norm(psi, x);
}

void solve_isw(vector<long double> &psi, vector<long double> &x, long double E){
    long double dx = x[1] - x[0];
    long double v;
    for(unsigned int j = 1; j<x.size()-1;j++){
        v = ((-0.5 * L < x[j]) && (x[j] < 0.5 * L)) ? 0.0 : Vinf;
        psi[j+1] = 2.0 * (pow(dx,2) * (v - E) + 1) * psi[j] - psi[j-1];
    }
    norm(psi, x);
}

void solve_fsw(vector<long double> &psi, vector<long double> &x, long double E){
    long double dx = x[1] - x[0];
    long double v;
    for(unsigned int j = 1; j<x.size()-1;j++){
        v = ((-0.5 * L < x[j]) && (x[j] < 0.5 * L)) ? 0.0 : V0;
        psi[j+1] = 2.0 * (pow(dx,2) * (v - E) + 1) * psi[j] - psi[j-1];
    }
    norm(psi, x);
}

void solve_h(vector<long double> &psi, vector<long double> &x, long double E){
    long double dx = x[1] - x[0];
    long double v;
    for(unsigned int j = 1; j<x.size()-1;j++){
        v = 0.5 * l * (l + 1) / pow(x[j],2) - 1 / x[j];
        psi[j+1] = 2.0 * (pow(dx,2) * (v - E) + 1) * psi[j] - psi[j-1];
    }
    norm(psi, x);
}

void initialize(vector<long double> &psi, vector<long double> &x, vector<long double> &E, struct Parameters &params){
    string helpmethod = "qho:  Quantum Harmonic Oscillator\n"
                        "isw:  Infinite Square Well   \n"
                        "fsw:  Finite Square Well     \n"
                        "H:    Hydrogen Atom     \n";

    cout << "Input a potential, type 'help' to get a list of available options"<<endl;
    cin >> params.method;
    cout << endl;
    if(params.method == "help"){
        cout << helpmethod << endl;
        return;
    }

    cout << "Input: inf"<<endl;
    cin >> params.inf;
    cout << endl;

    cout << "Input: xi xf N"<<endl;
    cin >> params.xi >> params.xf >> params.N;
    cout << endl;

    cout << "Input: Ei Ef N_E"<<endl;
    cin >> params.Ei >> params.Ef >> params.N_E;
    cout << endl;

    psi.resize(params.N);
    x.resize(params.N);
    E.resize(params.N_E);

    psi[0] = 0;
    psi[1] = params.dx;
    x[0] = params.xi;
    E[0] = params.Ei;

    params.dx = (params.xf-params.xi)/(long double)(params.N-1);
    params.dE = (params.Ef-params.Ei)/(long double)(params.N_E-1);

    for(unsigned int i=1;i<params.N;i++){
        x[i] = x[i-1] + params.dx;
    }

    for(unsigned int i=1;i<params.N_E;i++){
        E[i] = E[i-1] + params.dE;
    }
}

long double obj_func(void (*f)(vector<long double> &psi, vector<long double> &x, long double E),
        vector<long double> &psi, vector<long double> &x, long double e){
    f(psi, x, e);
    return psi.back();
}


int main() {
    vector<long double> psi, x, E;
    initialize(psi, x, E, params);
    long double a = 0.0;
    long double b = 10.0;
    long double c = 0.0;

    void (*f_ptr)(vector<long double> &, vector<long double> &, long double) = solve_qho;

    if(obj_func(f_ptr,psi,x,a)*obj_func(f_ptr,psi,x,b)<0.0){
        // bisection
        long double fa, fb, fc;

        while(abs(b-a)>eps){
            c = 0.5*(a+b);
            fa = obj_func(f_ptr,psi,x,a); fb = obj_func(f_ptr,psi,x,b); fc = obj_func(f_ptr,psi,x,c);

            if (abs(fa) < zero) { c = a; break; }
            if (abs(fb) < zero) { c = b; break; }
            if (abs(fc) < zero) { break; }

            if(fa*fc > 0) { a = c; }
            else if(fb*fc > 0) { b = c; }
            else { cout << "Uh, oh. Something's fishy" << endl; }
        }
    }
    else {
        // regula falsi
        long double fa, fb, fc;
        while (abs(b - a) > eps) {
            fa = obj_func(f_ptr, psi, x, a);
            fb = obj_func(f_ptr, psi, x, b);
            c = (a * fb - b * fa) / (fb - fa);
            fc = obj_func(f_ptr, psi, x, c);

            if (abs(fa) < zero) { c = a; break; }
            if (abs(fb) < zero) { c = b; break; }
            if (abs(fc) < zero) { break; }

            if (fa * fc > 0) { a = c; }
            else { b = c; }
        }
    }
    cout << c << endl;
return 0;
}
