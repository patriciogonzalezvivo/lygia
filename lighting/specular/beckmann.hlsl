#include "../common/beckmann.hlsl"

#ifndef FNC_SPECULAR_BECKMANN
#define FNC_SPECULAR_BECKMANN

float specularBeckmann(ShadingData shadingData) {
    return beckmann(shadingData.NoH, shadingData.roughness);
}

#endif