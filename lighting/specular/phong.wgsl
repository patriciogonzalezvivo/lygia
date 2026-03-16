#include "../../math/powFast.wgsl"
#include "../toShininess.wgsl"

// #define SPECULAR_POW(A,B) powFast(A,B)
// #define SPECULAR_POW(A,B) pow(A,B)

// https://github.com/glslify/glsl-specular-phong
fn specularPhong3(L: vec3f, N: vec3f, V: vec3f, shininess: f32) -> f32 {
    let R = reflect(L, N); // 2.0 * dot(N, L) * N - L;
    return SPECULAR_POW(max(0.0, dot(R, -V)), shininess);
}

fn specularPhong(shadingData: ShadingData) -> f32 {
    return specularPhong(shadingData.L, shadingData.N, shadingData.V, shadingData.roughness);
}

fn specularPhongRoughness(shadingData: ShadingData) -> f32 {
    return specularPhong(shadingData.L, shadingData.N, shadingData.V, toShininess(shadingData.roughness, 0.0));
}
