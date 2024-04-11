// eigen_example.h
#ifndef EIGEN_EXAMPLE_H
#define EIGEN_EXAMPLE_H

#include <Eigen/Dense>

Eigen::MatrixXd eigen3_exampleFunction() {
    Eigen::MatrixXd mat(2, 2);

    mat << 1, 2, 3, 4;

    return mat;
}

#endif // EIGEN_EXAMPLE_H

