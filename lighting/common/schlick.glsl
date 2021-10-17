#include "../../math/pow5.glsl"

#ifndef FNC_SCHLICK
#define FNC_SCHLICK

// Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
vec3 schlick(const vec3 f0, float f90, float VoH) {
    float f = pow5(1.0 - VoH);
    return f + f0 * (f90 - f);
}

vec3 schlick(vec3 f0, vec3 f90, float VoH) {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

float schlick(float f0, float f90, float VoH) {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

#endif