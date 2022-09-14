/*
original_author: Patricio Gonzalez Vivo
description: returns a 2x2 rotation matrix
use: rotate2d(<float> radians)
*/

#ifndef FNC_ROTATE2D
#define FNC_ROTATE2D
mat2 rotate2d(in float radians){
    float c = cos(radians);
    float s = sin(radians);
    return mat2(c, -s, s, c);
}
#endif
