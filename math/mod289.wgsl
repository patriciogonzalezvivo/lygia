/*
contributors: [Stefan Gustavson, Ian McEwan]
description: modulus of 289
use: 
    - mod289(x: f32) -> f32
    - mod289_2(x: vec2f) -> vec2f
    - mod289_3(x: vec3f) -> vec3f
    - mod289_4(x: vec4f) -> vec4f
*/

fn mod289(x: f32) -> f32 { return x - floor(x * (1. / 289.)) * 289.; }
fn mod289_2(x: vec2f) -> vec2f { return x - floor(x * (1. / 289.)) * 289.; }
fn mod289_3(x: vec3f) -> vec3f { return x - floor(x * (1. / 289.)) * 289.; }
fn mod289_4(x: vec4f) -> vec4f { return x - floor(x * (1. / 289.)) * 289.; }
