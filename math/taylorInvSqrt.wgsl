/*
contributors: [Stefan Gustavson, Ian McEwan]
description: Fast, accurate inverse square root. 
use:
    - taylorInvSqrt(r: f32) -> f32
    - taylorInvSqrt2(r: vec2f) -> vec2f
    - taylorInvSqrt3(r: vec3f) -> vec3f
    - taylorInvSqrt4(r: vec4f) -> vec4f
*/

fn taylorInvSqrt(r: f32) -> f32 { return 1.79284291400159 - 0.85373472095314 * r; }
fn taylorInvSqrt2(r: vec2f) -> vec2f { return 1.79284291400159 - 0.85373472095314 * r; }
fn taylorInvSqrt3(r: vec3f) -> vec3f { return 1.79284291400159 - 0.85373472095314 * r; }
fn taylorInvSqrt4(r: vec4f) -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }
