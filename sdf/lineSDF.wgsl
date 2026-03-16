#include "../math/saturate.wgsl"

/*
contributors: Inigo Quiles
description: Segment SDF
use: lineSDF(<vec2> st, <vec2> A, <vec2> B)
*/

fn lineSDF2(st: vec2f, a: vec2f, b: vec2f) -> f32 {
    let b_to_a = b - a;
    let to_a = st - a;
    let h = saturate(dot(to_a, b_to_a)/dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a );
}

fn lineSDF3(p: vec3f, a: vec3f, b: vec3f) -> f32 {
    //https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    return length(cross(p - a, p - b))/length(b - a);
}
