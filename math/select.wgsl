/*
contributors: Patricio Gonzalez Vivo
description: "Returns A when cond is true, and B otherwise. This is in part to bring a compatibility layer with WGSL \n"
use: <float|vec2|vec3|vec4> select(<float|vec2|vec3|vec4> A, <float|vec2|vec3|vec4> B, <bool> cond)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn select(A: i32, B: i32, cond: bool) -> i32 { return cond ? A : B; }
fn selecta(A: f32, B: f32, cond: bool) -> f32 { return cond ? A : B; }
fn select2(A: vec2f, B: vec2f, cond: bool) -> vec2f { return cond ? A : B; }
fn select3(A: vec3f, B: vec3f, cond: bool) -> vec3f { return cond ? A : B; }
fn select4(A: vec4f, B: vec4f, cond: bool) -> vec4f { return cond ? A : B; }
