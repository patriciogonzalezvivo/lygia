/*
original_author: Patricio Gonzalez Vivo
description: power of 5
use: pow5(<float|vec2|vec3|vec4> x)
*/

#ifndef FNC_POW5
#define FNC_POW5

float pow5(const in float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

vec2 pow5(const in vec2 x) {
    vec2 x2 = x * x;
    return x2 * x2 * x;
}

vec3 pow5(const in vec3 x) {
    vec3 x2 = x * x;
    return x2 * x2 * x;
}

vec4 pow5(const in vec4 x) {
    vec4 x2 = x * x;
    return x2 * x2 * x;
}

#endif
