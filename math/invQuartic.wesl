/*
contributors: Inigo Quiles
description: inverse quartic polynomial https://iquilezles.org/articles/smoothsteps/
*/

fn invQuartic(v: f32) -> f32 { return sqrt(1.0-sqrt(1.0-v)); }
fn invQuartic2(v: vec2f) -> vec2f { return sqrt(1.0-sqrt(1.0-v)); }
fn invQuartic3(v: vec3f) -> vec3f { return sqrt(1.0-sqrt(1.0-v)); }
fn invQuartic4(v: vec4f) -> vec4f { return sqrt(1.0-sqrt(1.0-v)); }
