#include "../math/const.glsl"

/*
description: generate the SDF of a icosahedron
use: <float> icosahedronSDF( in <vec3> pos, in <float> size ) 
*/

#ifndef FNC_ICOSAHEDRONSDF
#define FNC_ICOSAHEDRONSDF

float icosahedronSDF(vec3 p, float radius) {
    float q = 2.61803398875; // Golden Ratio + 1 = (sqrt(5)+3)/2;
    vec3 n1 = normalize(vec3(q, 1,0));
    vec3 n2 = vec3(0.57735026919);  // = sqrt(3)/3);

    p = abs(p / radius);
    float a = dot(p, n1.xyz);
    float b = dot(p, n1.zxy);
    float c = dot(p, n1.yzx);
    float d = dot(p, n2) - n1.x;
    return max(max(max(a,b),c)-n1.x,d) * radius;
}

#endif