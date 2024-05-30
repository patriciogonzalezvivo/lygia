/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 scale matrix
use: 
    - <mat2> scale2d(<float|vec2> radians)
    - <mat2> scale2d(<float> x, <float> y)
*/

#ifndef FNC_SCALE2D
mat2 scale2d(float s) { return mat2(s, 0.0, 0.0, s); }
mat2 scale2d(vec2 s) { return mat2(s.x, 0.0, 0.0, s.y); }
mat2 scale2d(float x, float y) { return mat2(x, 0.0, 0.0,  y); }
#endif