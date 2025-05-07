/*
contributors: Inigo Quiles
description: generate the SDF of a cylinder

*/

fn cylinderSDF(p: vec3f, h: vec2f) -> f32 {
    let d = abs(vec2f(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(vec2f(max(d.x, 0.0), max(d.y, 0.0)));
}