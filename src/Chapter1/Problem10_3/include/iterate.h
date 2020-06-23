//
// Created by ethan on 6/22/2020.
//

#ifndef PROBLEM10_3_ITERATE_H

#include <cmath>
#include <vector>

using namespace std;

#define PROBLEM10_3_ITERATE_H


long double iterate(vector<long double> &psi, vector<long double> &x,long double (*V)(long double X), long double E){
    long double dx = x[1] - x[0];
    long double dleft, dright;
    long int size = psi.size();
    long int back_left = size/2-1;
    long int front_right = size/2;

    //Solve left
    for(unsigned int j = 1; j<back_left;j++){
        psi[j+1] = 2.0 * (pow(dx,2) * (V(x[j]) - E) + 1) * psi[j] - psi[j-1];
    }
    //Solve right
    for(unsigned int j = size-2;j>front_right;j--){
        psi[j-1] = 2.0 * (pow(dx,2) * (V(x[j]) - E) + 1) * psi[j] - psi[j+1];
    }

    //Scale depending on which is larger
    if(psi.at(back_left)>psi.at(front_right)){
        for(unsigned int j=0;j<front_right;j++) {
            psi[j] = psi[j]*(psi.at(front_right)/psi.at(back_left));
        }
    }
    else if(psi.at(back_left)<psi.at(front_right)){
        for(unsigned int j=front_right;j<size;j++) {
            psi[j] = psi[j]*(psi.at(back_left)/psi.at(front_right));
        }
    }

    //Numerically calculate their derivatives
    dleft = (psi.at(back_left)-psi.at(back_left-1))/(x.at(back_left)-x.at(back_left-1));
    dright = (psi.at(front_right+1)-psi.at(front_right))/(x.at(front_right+1)-x.at(front_right));

    if(dleft > dright){ return 1; }
    else{ return -1; }
}

#endif //PROBLEM10_3_ITERATE_H
