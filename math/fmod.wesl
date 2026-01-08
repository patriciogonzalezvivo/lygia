/*
contributors: Patricio Gonzalez Vivo
description: Floored modulo that matches GLSL mod() behavior (uses floor, not truncation).
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn fmod2(x: vec2f, y: vec2f) -> vec2f { return x - y * floor(x / y); }
fn fmod3(x: vec3f, y: vec3f) -> vec3f { return x - y * floor(x / y); }
fn fmod4(x: vec4f, y: vec4f) -> vec4f { return x - y * floor(x / y); }
