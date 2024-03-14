/*
contributors: Inigo Quiles
description: inverse quartic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|vec2|vec3|vec4> invQuartic(<float|vec2|vec3|vec4> value);
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

fn invQuartic(v: f32) -> f32 { return sqrt(1.0-sqrt(1.0-v)); }
fn invQuartic2(v: vec2f) -> vec2f { return sqrt(1.0-sqrt(1.0-v)); }
fn invQuartic3(v: vec3f) -> vec3f { return sqrt(1.0-sqrt(1.0-v)); }
fn invQuartic4(v: vec4f) -> vec4f { return sqrt(1.0-sqrt(1.0-v)); }
