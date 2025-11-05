/*
contributors: Patricio Gonzalez Vivo
description: Flip Y axis
use: <vec2|vec3|vec4> flipY(<vec2|vec3|vec4> st)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn flipY2(v: vec2f) -> vec2f { return vec2f(v.x, 1.0 - v.y); }
fn flipY3(v: vec3f) -> vec3f { return vec3f(v.x, 1.0 - v.y, v.z); }
fn flipY4(v: vec4f) -> vec4f { return vec4f(v.x, 1.0 - v.y, v.z, v.w); }
