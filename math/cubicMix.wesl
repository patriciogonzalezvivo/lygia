import package::math::cubic::{cubic, cubic2, cubic3, cubic4};

/*
contributors: Patricio Gonzalez Vivo
description: cubic polynomial interpolation between two values
use: <float|vec2|vec3|vec4> cubicMix(<float|vec2|vec3|vec4> A, <float|vec2|vec3|vec4> B, float t)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

fn cubicMix(A: f32, B: f32, t: f32) -> f32 { return A + (B - A) * cubic(t); }
fn cubicMix2(A: vec2f, B: vec2f, t: vec2f) -> vec2f { return A + (B - A) * cubic2(t); }
fn cubicMix3(A: vec3f, B: vec3f, t: vec3f) -> vec3f { return A + (B - A) * cubic3(t); }
fn cubicMix4(A: vec4f, B: vec4f, t: vec4f) -> vec4f { return A + (B - A) * cubic4(t); }
