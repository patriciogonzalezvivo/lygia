/*
contributors:  Inigo Quiles
description: generate the SDF of a bounding box
use: <float> boxFrameSDF( <vec3> p, <vec3> b, <float> e )
*/

fn boxFrameSDF(p: vec3f, b: vec3f, e: f32) -> f32 {
    p = abs(p) - b;
    let q = abs(p + e) - e;

    return min(min(
        length(max(vec3f(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
        length(max(vec3f(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
        length(max(vec3f(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}
