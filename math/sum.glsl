/*
contributors: Patricio Gonzalez Vivo
description: Sum elements of a vector
use: <float> sum(<vec2|vec3|vec4> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SUM
#define FNC_SUM
float sum( float v ) { return v; }
float sum( vec2 v ) { return v.x+v.y; }
float sum( vec3 v ) { return v.x+v.y+v.z; }
float sum( vec4 v ) { return v.x+v.y+v.z+v.w; }
#endif
