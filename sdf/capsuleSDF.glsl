#include "../math/saturate.glsl"
/*
contributors:  Inigo Quiles
description: generate a SDF of a capsule
use: <float> capusleSDF( in <vec3> pos, in <vec3> a, <vec3> b, <float> r ) 
*/

#ifndef FNC_CAPSULESDF
#define FNC_CAPSULESDF

float capsuleSDF( vec3 p, vec3 a, vec3 b, float r ) {
    vec3 pa = p-a, ba = b-a;
    float h = saturate( dot(pa,ba)/dot(ba,ba) );
    return length( pa - ba*h ) - r;
}

#endif