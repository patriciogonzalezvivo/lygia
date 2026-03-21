#include "boxSDF.wgsl"

/*
description: generate the SDF of a cube
use: <float> cubeSDF( in <vec3> pos, in <float> size ) 
*/

fn cubeSDF(p: vec3f, s: f32) -> f32 { return boxSDF(p, vec3f(s)); }
