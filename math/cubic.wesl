/*
contributors: Inigo Quiles
description: cubic polynomial https://iquilezles.org/articles/smoothsteps/
*/

fn cubic(v: f32) -> f32 { return v*v*(3.0-2.0*v); }
fn cubic2(v: vec2f) -> vec2f { return v*v*(3.0-2.0*v); }
fn cubic3(v: vec3f) -> vec3f { return v*v*(3.0-2.0*v); }
fn cubic4(v: vec4f) -> vec4f { return v*v*(3.0-2.0*v); }