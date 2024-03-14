/*
contributors: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion 
use: 
    - decimate(v: f32, d:f32) -> f32
    - decimate2(v: vec2f, d:vec2f) -> vec2f
    - decimate3(v: vec3f, d:vec3f) -> vec3f
    - decimate4(v: vec4f, d:vec4f) -> vec4f
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

fn decimate(v: f32, d:f32) -> f32 { return floor(v * d) / d; }
fn decimate2(v: vec2f, d:vec2f) -> vec2f { return floor(v * d) / d; }
fn decimate3(v: vec3f, d:vec3f) -> vec3f { return floor(v * d) / d; }
fn decimate4(v: vec4f, d:vec4f) -> vec4f { return floor(v * d) / d; }