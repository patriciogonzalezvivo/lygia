#include "../../math/const.wgsl"

// https://github.com/glslify/glsl-specular-beckmann

fn beckmann(_NoH: f32, roughness: f32) -> f32 {
    let NoH = max(_NoH, 0.0001);
    let cos2Alpha = NoH * NoH;
    let tan2Alpha = (cos2Alpha - 1.0) / cos2Alpha;
    let roughness2 = roughness * roughness;
    let denom = PI * roughness2 * cos2Alpha * cos2Alpha;
    return exp(tan2Alpha / roughness2) / denom;
}
