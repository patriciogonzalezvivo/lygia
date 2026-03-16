/*
contributors:  Inigo Quiles
description: generate the SDF of a plane
use: <float> planeSDF( in <vec3> pos, in <vec2> h ) 
*/

fn planeSDF3(p: vec3f) -> f32 {
   return p.y; 
}

fn planeSDF3a(p: vec3f, planePoint: vec3f, planeNormal: vec3f) -> f32 {
    return (dot(planeNormal, p) + dot(planeNormal, planePoint)) / length(planeNormal);
}
