#include "rgb2hcv.wgsl"

fn rgb2hsl(rgb: vec3f) -> vec3f {
    let HCV = rgb2hcv(rgb);
    let L = HCV.z - HCV.y * 0.5;
    let S = HCV.y / (1.0 - abs(L * 2.0 - 1.0) + 1e-10);
    return vec3f(HCV.x, S, L);
}