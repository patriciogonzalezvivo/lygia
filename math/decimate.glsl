/*
contributors: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion 
use: decimate(<float|vec2|vec3|vec4> value, <float|vec2|vec3|vec4> presicion)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

#ifndef FNC_DECIMATE
#define FNC_DECIMATE
#define decimate(V, P) (floor(V * P)/P)
#endif