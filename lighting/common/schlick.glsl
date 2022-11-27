#include "../../math/pow5.glsl"

#ifndef FNC_SCHLICK
#define FNC_SCHLICK

// Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
vec3 schlick(const in vec3 f0, const in float f90, const in float VoH) {
    float f = pow5(1.0 - VoH);
    return f + f0 * (f90 - f);
}

vec3 schlick(const in vec3 f0, const in vec3 f90, const in float VoH) {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

float schlick(const in float f0, const in float f90, const in float VoH) {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

#endif