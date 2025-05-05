/*
contributors: Inigo Quiles
description: inverse cubic polynomial https://iquilezles.org/articles/smoothsteps/
*/

fn invCubic(v: f32) -> fn32 { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
fn invCubic2(v: vec2f) -> vec2f { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
fn invCubic3(v: vec3f) -> vec3f { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
fn invCubic4(v: vec4f) -> vec4f { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
