/*
contributors: Patricio Gonzalez Vivo
description: extend GLSL min function to add more arguments
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn mmin2(v: vec2f) -> f32 { return min(v.x, v.y); }
fn mmin3(v: vec3f) -> f32 { return mmin(v.x, v.y, v.z); }
fn mmin4(v: vec4f) -> f32 { return mmin(v.x, v.y, v.z, v.w); }
