/*
contributors: Ignacio Castaño
description: Debuging cube http://the-witness.net/news/2012/02/seamless-cube-map-filtering/
use: <vec3> debugCube(<vec3> _normal, <float> cube_size, <float> lod)
*/

#ifndef FNC_DEBUGCUBE
#define FNC_DEBUGCUBE

vec3 debugCube(const in vec3 v, const in float cube_size, const in float lod ) {
    float M = max(max(abs(v.x), abs(v.y)), abs(v.z));
    float scale = 1.0 - exp2(lod) / cube_size;
    if (abs(v.x) != M) v.x *= scale;
    if (abs(v.y) != M) v.y *= scale;
    if (abs(v.z) != M) v.z *= scale;
    return v;
}

#endif