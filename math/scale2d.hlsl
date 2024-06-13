/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 scale matrix
use:
    - <float2x2> scale2d(<float|float2> radians)
    - <float2x2> scale2d(<float> x, <float> y)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SCALE2D
float2x2 scale2d(float s) { return float2x2(s, 0.0, 0.0, s); }
float2x2 scale2d(float2 s) { return float2x2(s.x, 0.0, 0.0, s.y); }
float2x2 scale2d(float x, float y) { return float2x2(x, 0.0, 0.0,  y); }
#endif