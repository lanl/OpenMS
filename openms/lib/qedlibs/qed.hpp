#ifndef QED_MODULE_H
#define QED_MODULE_H

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <Eigen/Dense>


using namespace std; // for convenience

double gaussian_factor_vt_qedhf(double freq, double eta_p, double eta_q, double eta_r = -1, double eta_s = -1);

vector<double> test_gaussian(int N, double freq, vector<double>& eta);

void update_fock_energy_gradient_vt_qedhf(pybind11::object wf, Eigen::MatrixXd gvar);

#endif // QED_MODULE_H
