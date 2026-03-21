/*
contributors: NAN
description: |
    Returns a color ambient occlusion based on a pre-computed visibility term.
    The albedo term is meant to be the diffuse color or f0 for the diffuse and
    specular terms respectively.
use: <vec3> gtaoMultiBounce(<float> visibility, <vec3> albedo)
*/

fn gtaoMultiBounce(visibility: f32, albedo: vec3f) -> vec3f {
    // Jimenez et al. 2016, "Practical Realtime Strategies for Accurate Indirect Occlusion"
    let a = 2.0404 * albedo - 0.3324;
    let b = -4.7951 * albedo + 0.6417;
    let c = 2.7552 * albedo + 0.6903;

    return max(vec3f(visibility), ((visibility * a + b) * visibility + c) * visibility);
}
