/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 rotation matrix
use: <float2x2> rotate2d(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE2D
#define FNC_ROTATE2D
float2x2 rotate2d(const in float r){
    float c = cos(r);
    float s = sin(r);
    return float2x2(c, -s, s, c);
}
#endif
