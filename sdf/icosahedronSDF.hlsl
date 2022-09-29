#include "../math/const.hlsl"

/*
description: generate the SDF of a icosahedron
use: <float> icosahedronSDF( in <float3> pos, in <float> size ) 
*/

#ifndef FNC_ICOSAHEDRONSDF
#define FNC_ICOSAHEDRONSDF

float icosahedronSDF(float3 p, float radius) {
    float q = 2.61803398875; // Golden Ratio + 1 = (sqrt(5)+3)/2;
    float3 n1 = normalize(float3(q, 1,0));
    float3 n2 = float3(0.57735026919, 0.57735026919, 0.57735026919);  // = sqrt(3)/3);

    p = abs(p / radius);
    float a = dot(p, n1.xyz);
    float b = dot(p, n1.zxy);
    float c = dot(p, n1.yzx);
    float d = dot(p, n2) - n1.x;
    return max(max(max(a,b),c)-n1.x,d) * radius;
}

#endif