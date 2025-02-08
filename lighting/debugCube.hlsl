/*
contributors: Ignacio Castaño
description: Debugging cube http://the-witness.net/news/2012/02/seamless-cube-map-filtering/
use: <float3> debugCube(<float3> _norma, <float> cube_size, <float> lod)
*/

#ifndef FNC_DEBUGCUBE
#define FNC_DEBUGCUBE

float3 debugCube( float3 v, float cube_size, float lod ) {
    float M = max(max(abs(v.x), abs(v.y)), abs(v.z));
    float scale = 1.0 - exp2(lod) / cube_size;
    if (abs(v.x) != M) v.x *= scale;
    if (abs(v.y) != M) v.y *= scale;
    if (abs(v.z) != M) v.z *= scale;
    return v;
}

#endif