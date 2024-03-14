/*
contributors: Patricio Gonzalez Vivo
description: extend GLSL Max function to add more arguments
*/

fn mmax2(v: vec2f) -> f32 { return max(v.x, v.y); }
fn mmax3(v: vec3f) -> f32 { return mmax(v.x, v.y, v.z); }
fn mmax4(v: vec4f) -> f32 { return mmax(v.x, v.y, v.z, v.w); }
