/*
contributors: Inigo Quiles
description: quartic polynomial https://iquilezles.org/articles/smoothsteps/
use:
    - quartic(v: f32) -> f32
    - quartic2(v: vec2f) -> vec2f
    - quartic3(v: vec3f) -> vec3f
    - quartic4(v: vec4f) -> vec4f
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

fn quartic(v: f32) -> f32 { return v*v*(2.0-v*v); }
fn quartic2(v: vec2f) -> vec2f { return v*v*(2.0-v*v); }
fn quartic3(v: vec3f) -> vec3f { return v*v*(2.0-v*v); }
fn quartic4(v: vec4f) -> vec4f { return v*v*(2.0-v*v); }