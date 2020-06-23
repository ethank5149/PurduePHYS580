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

const long double Vinf = 1000.0;
const long double L = 10.0;
const long double zero = 1.0e-10;
const long double eps = 1.0e-8;

struct Parameters{
    int N = 1000;
    long double Ei = 0.0;
    long double Ef = 10.0;
    long double dx = 2.0/(long double)(N-1);
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

long double obj_func(vector<long double> &psi, vector<long double> &x, long double E){
    long double dx = x[1] - x[0];
    long double v;
    for(unsigned int j = 1; j<x.size()-1;j++){
        v = ((-0.5 * L < x[j]) && (x[j] < 0.5 * L)) ? 0.0 : Vinf;
        psi[j+1] = 2.0 * (pow(dx,2) * (v - E) + 1) * psi[j] - psi[j-1];
    }
    norm(psi, x);
    return psi.back();
}

void initialize(vector<long double> &psi, vector<long double> &x, struct Parameters &params){
    cout << "Input: N Ei Ef"<<endl;
    cin >> params.N >> params.Ei >> params.Ef;
    cout << endl;

    psi.resize(params.N);
    x.resize(params.N);

    params.dx = 2.0/(long double)(params.N-1);

    psi[0] = 0;
    psi[1] = params.dx;
    x[0] = -1.0;

    for(unsigned int i=1;i<params.N;i++){
        x[i] = x[i-1] + params.dx;
    }
}

int main() {
    vector<long double> psi, x, E;
    initialize(psi, x, params);
    long double a = params.Ei;
    long double b = params.Ef;
    long double c = 0.0;

    if(obj_func(psi,x,a)*obj_func(psi,x,b)<0.0){
        // bisection
        long double fa, fb, fc;

        while(abs(b-a)>eps){
            c = 0.5*(a+b);
            fa = obj_func(psi,x,a); fb = obj_func(psi,x,b); fc = obj_func(psi,x,c);

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
            fa = obj_func(psi, x, a);
            fb = obj_func(psi, x, b);
            c = (a * fb - b * fa) / (fb - fa);
            fc = obj_func(psi, x, c);

            if (abs(fa) < zero) { c = a; break; }
            if (abs(fb) < zero) { c = b; break; }
            if (abs(fc) < zero) { break; }

            if (fa * fc > 0) { a = c; }
            else { b = c; }
        }
    }
    cout << setprecision(numeric_limits<long double>::digits10 + 1);
    cout << "E = " << c << endl;
    cout << "Checking Answer..." << endl;
    cout << "Is this close enough to an integer? -> " << 2.0*sqrt(2.0*c)/M_PI << endl;
    cout << "If so, You're good to go!" << endl;
return 0;
}

//Results

//N = 40
////n = 1: 1.233033575117588043 | -0.05406295867405493 % Error
////n = 2: 4.92413763515651226  | -0.21610927763203222 % Error
////n = 3: 11.04937392374371668 | -0.4857204924004033  % Error
////n = 4: 19.56901795417070389 | -0.8621969082632495  % Error
////n = 5: 30.42781655676662922 | -1.3445635461274381  % Error
////n = 6: 43.55534624168649316 | -1.931572552010588   % Error

//N = 200
////n = 1: 1.233674921095371246 | -0.0020774117994576476 % Error
////n = 2: 4.934392256662249565 | -0.008307199878937062  % Error
////n = 3: 11.10122967511415482 | -0.018690616176801584  % Error
////n = 4: 19.73265030235052109 | -0.0332257482755388    % Error
////n = 5: 30.82650300115346909 | -0.05191130780968817   % Error
////n = 6: 44.3800229262560606  | -0.07474548972553445   % Error

//N = 400
////n = 1: 1.233694173395633698 | -0.0005168791191142531 % Error
////n = 2: 4.934700218960642815 | -0.00206657896085329   % Error
////n = 3: 11.10278870910406113 | -0.004649445581606569  % Error
////n = 4: 19.73757722228765488 | -0.008265680288467398  % Error
////n = 5: 30.83853048086166382 | -0.012914876441096497  % Error
////n = 6: 44.4049602784216404  | -0.01859699998503756   % Error

//Analytic
////n = 1: 1.2337005501361697
////n = 2: 4.934802200544679
////n = 3: 11.103304951225528
////n = 4: 19.739208802178716
////n = 5: 30.842513753404244
////n = 6: 44.41321980490211