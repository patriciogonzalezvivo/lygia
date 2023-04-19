/*
original_author: Patricio Gonzalez Vivo
description: gaussian coeficient
use: <vec4|vec3|vec2|float> gaussian(<float> sigma, <vec4|vec3|vec2|float> d)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_gaussian.frag
*/

#ifndef FNC_GAUSSIAN
#define FNC_GAUSSIAN

float gaussian(float d, float sigma) { return exp(-(d*d) / (2.0 * sigma*sigma)); }
float gaussian( vec2 d, float sigma) { return exp(-( d.x*d.x + d.y*d.y) / (2.0 * sigma*sigma)); }
float gaussian( vec3 d, float sigma) { return exp(-( d.x*d.x + d.y*d.y + d.z*d.z ) / (2.0 * sigma*sigma)); }
float gaussian( vec4 d, float sigma) { return exp(-( d.x*d.x + d.y*d.y + d.z*d.z + d.w*d.w ) / (2.0 * sigma*sigma)); }

#endif