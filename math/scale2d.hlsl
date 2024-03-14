/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 scale matrix
use: 
    - <float2x2> scale2d(<float|float2> radians)
    - <float2x2> scale2d(<float> x, <float> y)
*/

#ifndef FNC_SCALE2D
float2x2 scale2d(float s) { return float2x2(s, 0.0, 0.0, s); }
float2x2 scale2d(float2 s) { return float2x2(s.x, 0.0, 0.0, s.y); }
float2x2 scale2d(float x, float y) { return float2x2(x, 0.0, 0.0,  y); }
#endif