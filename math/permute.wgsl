#include "mod289.wgsl"

fn permute(x: f32) -> f32 { return mod289_1(((x * 34.0) + 1.0) * x); }
fn permute2(x: vec2f) -> vec2f { return mod289_2(((x * 34.0) + 1.0) * x); }
fn permute3(x: vec3f) -> vec3f { return mod289_3(((x * 34.0) + 1.0) * x); }
fn permute4(x: vec4f) -> vec4f { return mod289_4(((x * 34.0) + 1.0) * x); }
