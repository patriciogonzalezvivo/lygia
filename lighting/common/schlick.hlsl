#include "../../math/pow5.hlsl"

#ifndef FNC_SCHLICK
#define FNC_SCHLICK

// Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
float3 schlick(const float3 f0, float f90, float VoH) {
    float f = pow5(1.0 - VoH);
    return f + f0 * (f90 - f);
}

float3 schlick(float3 f0, float3 f90, float VoH) {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

float schlick(float f0, float f90, float VoH) {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

#endif