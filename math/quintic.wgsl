/*
contributors: Inigo Quiles
description: quintic polynomial https://iquilezles.org/articles/smoothsteps/
*/

fn quintic(v: f32) -> f32 { return v*v*v*(v*(v*6.0-15.0)+10.0); }
fn quintic2(v: vec2f) -> vec2f { return v*v*v*(v*(v*6.0-15.0)+10.0); }
fn quintic3(v: vec3f) -> vec3f { return v*v*v*(v*(v*6.0-15.0)+10.0); }
fn quintic4(v: vec4f) -> vec4f { return v*v*v*(v*(v*6.0-15.0)+10.0); }
