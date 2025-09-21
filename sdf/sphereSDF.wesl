/*
contributors: Inigo Quiles
description: generate the SDF of a sphere

*/

fn sphereSDF(p: vec3f, s: f32) -> f32 { 
    return length(p) - s; 
}
