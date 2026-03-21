#include "../math/const.wgsl"

/*
contributors: Inigo Quiles
description:  Procedural generation of color palette algorithm explained here http://www.iquilezles.org/www/articles/palettes/palettes.htm
use: <vec3|vec4> palette(<float> t, <vec3|vec4> a, <vec3|vec4> b, <vec3|vec4> c, <vec3|vec4> d)
*/

fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f { return a + b * cos(TAU * ( c * t + d )); }
fn palettea(t: f32, a: vec4f, b: vec4f, c: vec4f, d: vec4f) -> vec4f { return a + b * cos(TAU * ( c * t + d )); }
