#include "point.glsl"
#include "pointEvaluate.glsl"
#include "directional.glsl"
#include "directionalEvaluate.glsl"
#include "../material.glsl"

#ifndef FNC_LIGHT_RESOLVE
#define FNC_LIGHT_RESOLVE

void lightResolve(LightPoint L, Material mat, inout ShadingData shadingData) {
    lightPointEvaluate(L, mat, shadingData);
}


void lightResolve(LightDirectional L, Material mat, inout ShadingData shadingData) {
    lightDirectionalEvaluate(L, mat, shadingData);
}

#endif