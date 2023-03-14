/*
original_author: Patricio Gonzalez Vivo
description: gaussian coeficient
use: <vec4|vec3|vec2|float> gaussian(<float> sigma, <vec4|vec3|vec2|float> d)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

#ifndef FNC_GAUSSIAN
#define FNC_GAUSSIAN

float gaussian(float sigma, float d) { return exp(-(d*d) / (2.0 * sigma*sigma)); }
float gaussian(float sigma, vec2 d) { return exp(-( d.x*d.x + d.y*d.y) / (2.0 * sigma*sigma)); }
float gaussian(float sigma, vec3 d) { return exp(-( d.x*d.x + d.y*d.y + d.z*d.z ) / (2.0 * sigma*sigma)); }
float gaussian(float sigma, vec4 d) { return exp(-( d.x*d.x + d.y*d.y + d.z*d.z + d.w*d.w ) / (2.0 * sigma*sigma)); }

#endif