#include "../math/const.glsl"

/*
description: generate the SDF of a dodecahedron
use: <float> dodecahedronSDF( in <vec3> pos [, in <float> size] ) 
*/

#ifndef FNC_DODECAHEDRONSDF
#define FNC_DODECAHEDRONSDF

float dodecahedronSDF(vec3 p) {
    vec3 n = normalize(vec3(PHI,1.0,0.0));
    p = abs(p);
    float a = dot(p,n.xyz);
    float b = dot(p,n.zxy);
    float c = dot(p,n.yzx);
    // return max(max(a,b),c)-PHI*n.y;
    return (max(max(a,b),c)-n.x);
}

float dodecahedronSDF(vec3 p, float radius) {
    vec3 n = normalize(vec3(PHI,1.0,0.0));

    p = abs(p / radius);
    float a = dot(p, n.xyz);
    float b = dot(p, n.zxy);
    float c = dot(p, n.yzx);
    return (max(max(a,b),c)-n.x) * radius;
}

#endif