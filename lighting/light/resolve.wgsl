#include "point.wgsl"
#include "pointEvaluate.wgsl"
#include "directional.wgsl"
#include "directionalEvaluate.wgsl"
#include "../material.wgsl"

fn lightResolve(L: LightPoint, mat: Material, shadingData: ShadingData) {
    lightPointEvaluate(L, mat, shadingData);
}

fn lightResolvea(L: LightDirectional, mat: Material, shadingData: ShadingData) {
    lightDirectionalEvaluate(L, mat, shadingData);
}
