/*
contributors: Inigo Quiles
description: generate the SDF of a torus

*/

fn torusSDF(p: vec3f, t: vec2f) -> f32 {
    return length( vec2f(length(p.xz)-t.x, p.y) )-t.y;
}

fn torusSDF4(p: vec3f, sc: vec2f, ra: f32, rb: f32) -> f32 {
    var pos = p;
    pos.x = abs(pos.x);
    let k = select(length(pos.xy), dot(pos.xy, sc), sc.y*pos.x > sc.x*pos.y);
    return sqrt( dot(pos, pos) + ra*ra - 2.0*ra*k ) - rb;
}
