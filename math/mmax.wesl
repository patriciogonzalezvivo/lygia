/*
contributors: Patricio Gonzalez Vivo
description: extend GLSL Max function to add more arguments
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn mmax2(v: vec2f) -> f32 { return max(v.x, v.y); }
fn mmax3(v: vec3f) -> f32 { return max(max(v.x, v.y), v.z); }
fn mmax4(v: vec4f) -> f32 { return max(max(max(v.x, v.y), v.z), v.w); }
