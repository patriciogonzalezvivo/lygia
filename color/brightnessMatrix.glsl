/*
contributors: Patricio Gonzalez Vivo
description: generate a matrix to change a the brightness of any color
use: brightnessMatrix(<float> amount)
*/

#ifndef FNC_BRIGHTNESSMATRIX
#define FNC_BRIGHTNESSMATRIX
mat4 brightnessMatrix(in float a) {
    return mat4(  1., 0., 0., 0.,
                  0., 1., 0., 0.,
                  0., 0., 1., 0.,
                  a, a, a, 1. );
}
#endif
