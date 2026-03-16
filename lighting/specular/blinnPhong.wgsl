#include "../../math/powFast.wgsl"
#include "../toShininess.wgsl"

// #define SPECULAR_POW(A,B) powFast(A,B)
// #define SPECULAR_POW(A,B) pow(A,B)

// https://github.com/glslify/glsl-specular-blinn-phong
fn specularBlinnPhong(NoH: f32, shininess: f32) -> f32 {
    return SPECULAR_POW(max(0.0, NoH), shininess);
}

fn specularBlinnPhonga(shadingData: ShadingData) -> f32 {
    return specularBlinnPhong(shadingData.NoH, shadingData.roughness);
}

fn specularBlinnPhongRoughness(shadingData: ShadingData) -> f32 {
    return specularBlinnPhong(shadingData.NoH, toShininess(shadingData.roughness, 0.0));
}
