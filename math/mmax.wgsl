/*
contributors: Patricio Gonzalez Vivo
description: extend GLSL Max function to add more arguments
use: 
    - mmax2(vec2f) -> f32
    - mmax3(vec3f) -> f32
    - mmax4(vec4f) -> f32
*/

fn mmax2(v: vec2f) -> f32 { return max(v.x, v.y); }
fn mmax3(v: vec3f) -> f32 { return mmax(v.x, v.y, v.z); }
fn mmax4(v: vec4f) -> f32 { return mmax(v.x, v.y, v.z, v.w); }
