/*
contributors: Matt DesLauriers
description: Performs a smoothstep using standard derivatives for anti-aliased edges at any level of magnification. From https://github.com/glslify/glsl-aastep
*/

fn aastep(threshold: f32, value: f32) -> f32 {
    let afwidth = 0.7 * fwidth(value);
    return smoothstep(threshold - afwidth, threshold + afwidth, value);
}