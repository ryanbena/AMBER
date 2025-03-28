#pragma once
#ifndef KERNEL_H
#define KERNEL_H

#include <vector>

namespace Kernel{
    int Poisson(float *h, const float *f, const float *b, const float relTol, const float N);
}

#endif