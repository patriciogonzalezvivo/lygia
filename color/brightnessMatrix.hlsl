/*
contributors: Patricio Gonzalez Vivo
description: generate a matrix to change a the brightness of any color
use: brightnessMatrix(<float> amount)
*/

#ifndef FNC_BRIGHTNESSMATRIX
#define FNC_BRIGHTNESSMATRIX
float4x4 brightnessMatrix(in float amount) {
    return float4x4(  1., 0., 0., amount,
                      0., 1., 0., amount,
                      0., 0., 1., amount,
                      0., 0., 0., 1. );
}
#endif
