/*
contributors: Patricio Gonzalez Vivo
description: fast approximation to pow()
use: 
    - powFast(a: f32, b: f32) -> f32
    - powFast2(a: vec2f, b: vec2f) -> vec2f
    - powFast3(a: vec3f, b: vec3f) -> vec3f
    - powFast4(a: vec4f, b: vec4f) -> vec4f
*/

fn powFast(a: f32, b: f32) -> f32 { return a / ((1.0 - b) * a + b); }
fn powFast2(a: vec2f, b: vec2f) -> vec2f { return a / ((1.0 - b) * a + b); }
fn powFast3(a: vec3f, b: vec3f) -> vec3f { return a / ((1.0 - b) * a + b); }
fn powFast4(a: vec4f, b: vec4f) -> vec4f { return a / ((1.0 - b) * a + b); }