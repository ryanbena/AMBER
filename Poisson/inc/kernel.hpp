#pragma once
#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include <Eigen/Core>
#include "Types.h"

namespace Kernel{
    void Poisson(Eigen::MatrixXd &h, Eigen::MatrixXd &f, Eigen::MatrixXd &b, const int& num_iter);
}

#endif