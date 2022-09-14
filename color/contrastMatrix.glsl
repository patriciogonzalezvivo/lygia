/*
original_author: Patricio Gonzalez Vivo
description: generate a matrix to change a the contrast of any color
use: contrastMatrix(<float> amount)
*/

#ifndef FNC_CONTRASTMATRIX
#define FNC_CONTRASTMATRIX
mat4 contrastMatrix(in float amount) {
    float t = ( 1. - amount ) * .5;
    return mat4( amount, .0, .0, .0,
                .0, amount, .0, .0,
                .0, .0, amount, .0,
                t,   t,      t, 1. );
}
#endif
