/*
contributors: Inigo Quiles
description: generate the SDF of a box

*/

fn boxSDF( p: vec3f, b: vec3f ) -> f32 {
    let d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(vec3f(max(d.x, 0.0), max(d.y, 0.0), max(d.z, 0.0)));
}
