/*
contributors: Patricio Gonzalez Vivo
description: generate a matrix to change a the contrast of any color
use: contrastMatrix(<float> amount)
*/

#ifndef FNC_CONTRASTMATRIX
#define FNC_CONTRASTMATRIX
mat4 contrastMatrix(in float a) {
    float t = ( 1. - a ) * .5;
    return mat4( a, .0, .0, .0,
                .0, a, .0, .0,
                .0, .0, a, .0,
                t,   t, t, 1. );
}
#endif
