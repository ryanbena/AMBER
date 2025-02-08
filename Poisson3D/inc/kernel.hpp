#pragma once
#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include "Types.h"

namespace Kernel{
    int Poisson(double *h, const double *f, const double *b, const double& relTol, const int& rows, const int& cols, const double& cell_size);
}

#endif