/*
contributors: Patricio Gonzalez Vivo
description: extend GLSL min function to add more arguments
*/

fn mmin2(v: vec2f) -> f32 { return min(v.x, v.y); }
fn mmin3(v: vec3f) -> f32 { return mmin(v.x, v.y, v.z); }
fn mmin4(v: vec4f) -> f32 { return mmin(v.x, v.y, v.z, v.w); }
