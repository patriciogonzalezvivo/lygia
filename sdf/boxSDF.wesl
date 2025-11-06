/*
contributors: Inigo Quiles
description: generate the SDF of a box

*/

fn boxSDF( p: vec3f ) -> f32 {
    let d = abs(p);
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,vec3f(0.0)));
}

fn boxSDF1( p: vec3f, b: vec3f ) -> f32 {
    let d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,vec3f(0.0)));
}
