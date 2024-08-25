#include "point.glsl"
#include "directional.glsl"
#include "../material.glsl"

#ifndef FNC_LIGHT_RESOLVE
#define FNC_LIGHT_RESOLVE

void lightResolve(LightPoint L, Material mat, inout ShadingData shadingData) {
    lightPoint(L, mat, shadingData);
}


void lightResolve(LightDirectional L, Material mat, inout ShadingData shadingData) {
    lightDirectional(L, mat, shadingData);
}

#endif