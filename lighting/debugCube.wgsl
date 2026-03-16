/*
contributors: Ignacio Castaño
description: Debugging cube http://the-witness.net/news/2012/02/seamless-cube-map-filtering/
use: <vec3> debugCube(<vec3> _normal, <float> cube_size, <float> lod)
*/

fn debugCube(v: vec3f, cube_size: f32, lod: f32) -> vec3f {
    let M = max(max(abs(v.x), abs(v.y)), abs(v.z));
    let scale = 1.0 - exp2(lod) / cube_size;
    if (abs(v.x) != M) v.x *= scale;
    if (abs(v.y) != M) v.y *= scale;
    if (abs(v.z) != M) v.z *= scale;
    return v;
}
