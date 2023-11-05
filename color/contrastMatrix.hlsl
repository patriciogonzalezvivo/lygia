/*
contributors: Patricio Gonzalez Vivo
description: generate a matrix to change a the contrast of any color
use: contrastMatrix(<float> amount)
*/

#ifndef FNC_CONTRASTMATRIX
#define FNC_CONTRASTMATRIX
float4x4 contrastMatrix(in float amount) {
    float t = ( 1. - amount ) * .5;
    return float4x4(  amount, .0, .0, t,
                      .0, amount, .0, t,
                      .0, .0, amount, t,
                      .0, .0,     .0, 1. );
}
#endif
