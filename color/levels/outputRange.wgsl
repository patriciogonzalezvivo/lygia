/*
contributors: Johan Ismael
description: |
    Color output range adjustment similar to Levels adjustment in Photoshop
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsOutputRange(<vec3|vec4> color, float minOutput, float maxOutput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn levelsOutputRange3(v: vec3f, oMin: vec3f, oMax: vec3f) -> vec3f { return mix(oMin, oMax, v); }
fn levelsOutputRange4(v: vec4f, oMin: vec3f, oMax: vec3f) -> vec4f { return vec4f(levelsOutputRange(v.rgb, oMin, oMax), v.a); }
fn levelsOutputRange3a(v: vec3f, oMin: f32, oMax: f32) -> vec3f { return levelsOutputRange(v, vec3f(oMin), vec3f(oMax)); }
fn levelsOutputRange4a(v: vec4f, oMin: f32, oMax: f32) -> vec4f { return vec4f(levelsOutputRange(v.rgb, oMin, oMax), v.a); }
