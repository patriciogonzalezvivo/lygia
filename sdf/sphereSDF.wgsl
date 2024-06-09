/*
contributors: Inigo Quiles
description: generate the SDF of a sphere
license: null
*/

fn sphereSDF(p: vec3f, s: f32) -> f32 { 
    return length(p) - s; 
}