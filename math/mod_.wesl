/*
contributors: Patricio Gonzalez Vivo
description: An implementation of mod that matches the GLSL mod.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn mod2(x: vec2f, y: vec2f) -> vec2f { return x - y * floor(x / y); }
fn mod3(x: vec3f, y: vec3f) -> vec3f { return x - y * floor(x / y); }
fn mod4(x: vec4f, y: vec4f) -> vec4f { return x - y * floor(x / y); }
