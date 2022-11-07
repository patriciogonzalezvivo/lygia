
/*
original_author: Johan Ismael
description: Similar to step but for an interval instead of a threshold. Returns 1 is x is between left and right, 0 otherwise
use: within(<float> minVal, <float|vec2|vec3|vec4> maxVal, <float|vec2|vec3|vec4> x)
*/

#ifndef FNC_WITHIN
#define FNC_WITHIN
float within(in float minVal, in float maxVal, in float x) {
    return step(minVal, x) * (1. - step(maxVal, x));
}

float within(in vec2 minVal, in vec2 maxVal, in vec2 x) {
    vec2 rta = step(minVal, x) * (1. - step(maxVal, x));
    return rta.x * rta.y;
}

float within(in vec3 minVal, in vec3 maxVal, in vec3 x) {
    vec3 rta = step(minVal, x) * (1. - step(maxVal, x));
    return rta.x * rta.y * rta.z;
}

float within(in vec4 minVal, in vec4 maxVal, in vec4 x) {
    vec4 rta = step(minVal, x) * (1. - step(maxVal, x));
    return rta.x * rta.y * rta.z * rta.w;
}
#endif