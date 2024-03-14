#include "mod289.wgsl"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: permute
use: 
    - permute(x: f32) -> f32
    - permute2(x: vec2f) -> vec2f
    - permute3(x: vec3f) -> vec3f
    - permute4(x: vec4f) -> vec4f
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

fn permute(x: f32) -> f32 { return mod289(((x * 34.0) + 1.0) * x); }
fn permute2(x: vec2f) -> vec2f { return mod289_2(((x * 34.0) + 1.0) * x); }
fn permute3(x: vec3f) -> vec3f { return mod289_3(((x * 34.0) + 1.0) * x); }
fn permute4(x: vec4f) -> vec4f { return mod289_4(((x * 34.0) + 1.0) * x); }
