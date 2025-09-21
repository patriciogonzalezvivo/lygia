/*
contributors: Johan Ismael
description: Map a v between one range to another.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn map(v: f32, iMin: f32, iMax: f32, oMin: f32, oMax: f32) -> f32 { return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin); }
fn map2(v: vec2f, iMin: vec2f, iMax: vec2f, oMin: vec2f, oMax: vec2f) -> vec2f { return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin); }
fn map3(v: vec3f, iMin: vec3f, iMax: vec3f, oMin: vec3f, oMax: vec3f) -> vec3f { return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin); }
fn map4(v: vec4f, iMin: vec4f, iMax: vec4f, oMin: vec4f, oMax: vec4f) -> vec4f { return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin); }
