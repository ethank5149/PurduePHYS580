//
// Created by ethan on 6/22/2020.
//

#ifndef PROBLEM10_2_STORE_H

#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

#define PROBLEM10_2_STORE_H

void store(vector<long double> &psi, vector<long double> &x) {
    ofstream file("../data/output.dat");
    file << setprecision(numeric_limits<long double>::digits10 + 1);
    file << R"(# x \psi)" << endl;
    for (unsigned int i = 0; i < psi.size(); i++) {
        file << x[i] << " " << pow(psi[i],2) << endl;
    }
}

#endif //PROBLEM10_2_STORE_H
