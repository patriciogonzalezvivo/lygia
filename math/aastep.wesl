/*
contributors: Matt DesLauriers
description: Performs a smoothstep using standard derivatives for anti-aliased edges at any level of magnification. From https://github.com/glslify/glsl-aastep
*/

fn aastep(threshold: f32, value: f32) -> f32 {
    // Use Euclidean length (L2 norm) to match GLSL, not fwidth (L1 norm)
    let afwidth = 0.7 * length(vec2f(dpdx(value), dpdy(value)));
    return smoothstep(threshold - afwidth, threshold + afwidth, value);
}