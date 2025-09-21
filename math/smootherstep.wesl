import package::math::quintic::{quintic, quintic2, quintic3, quintic4};

/*
contributors: Patricio Gonzalez Vivo
description: quintic polynomial step function
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn smootherstep(a: f32, b: f32, v: f32) -> f32 { return quintic( saturate( (v - a)/(b - a) )); }
fn smootherstep2(a: vec2f, b: vec2f, v: vec2f) -> vec2f { return quintic2( saturate( (v - a)/(b - a) )); }
fn smootherstep3(a: vec3f, b: vec3f, v: vec3f) -> vec3f { return quintic3( saturate( (v - a)/(b - a) )); }
fn smootherstep4(a: vec4f, b: vec4f, v: vec4f) -> vec4f { return quintic4( saturate( (v - a)/(b - a) )); }
