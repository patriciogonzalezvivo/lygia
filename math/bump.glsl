#include "saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: bump in a range between -1 and 1
use: <float|vec3> bump(<float|vec3> x[], <float|vec3> k])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_BUMP
#define FNC_BUMP

float bump(float x, float k){ return saturate( (1.0 - x * x) - k); }
vec3 bump(vec3 x, vec3 k){ return saturate( (1.0 - x * x) - k); }
float bump(float x) { return max(1.0 - x * x, 0.0); }
vec3 bump(vec3 x) { return max(vec3(1.,1.,1.) - x * x, vec3(0.,0.,0.)); }

#endif