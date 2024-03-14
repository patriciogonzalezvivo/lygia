/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 rotation matrix
use: <mat2> rotate2d(<float> radians)
*/

#ifndef FNC_ROTATE2D
#define FNC_ROTATE2D
mat2 rotate2d(const in float r){
    float c = cos(r);
    float s = sin(r);
    return mat2(c, -s, s, c);
}
#endif
