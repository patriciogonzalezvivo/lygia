/*
original_author: Inigo Quiles
description: cubic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|vec2|vec3|vec4> cubic(<float|vec2|vec3|vec4> value[, <float> in, <float> out]);
*/

#ifndef FNC_CUBIC
#define FNC_CUBIC 
float cubic(const in float v) { return v*v*(3.0-2.0*v); }
vec2  cubic(const in vec2 v)  { return v*v*(3.0-2.0*v); }
vec3  cubic(const in vec3 v)  { return v*v*(3.0-2.0*v); }
vec4  cubic(const in vec4 v)  { return v*v*(3.0-2.0*v); }

float cubic(const in float value, in float slope0, in float slope1) {
    float a = slope0 + slope1 - 2.;
    float b = -2. * slope0 - slope1 + 3.;
    float c = slope0;
    float value2 = value * value;
    float value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

vec2 cubic(const in vec2 value, in float slope0, in float slope1) {
    float a = slope0 + slope1 - 2.;
    float b = -2. * slope0 - slope1 + 3.;
    float c = slope0;
    vec2 value2 = value * value;
    vec2 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

vec3 cubic(const in vec3 value, in float slope0, in float slope1) {
    float a = slope0 + slope1 - 2.;
    float b = -2. * slope0 - slope1 + 3.;
    float c = slope0;
    vec3 value2 = value * value;
    vec3 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}

vec4 cubic(const in vec4 value, in float slope0, in float slope1) {
    float a = slope0 + slope1 - 2.;
    float b = -2. * slope0 - slope1 + 3.;
    float c = slope0;
    vec4 value2 = value * value;
    vec4 value3 = value * value2;
    return a * value3 + b * value2 + c * value;
}
#endif