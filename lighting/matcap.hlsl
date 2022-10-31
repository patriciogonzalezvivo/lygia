/*
original_author: Patricio Gonzalez Vivo
description: matcap
use: <float2> matcap(<float3> eye, <float3> normal) 
*/

#ifndef FNC_MATCAP
#define FNC_MATCAP
float2 matcap(float3 eye, float3 normal) {
    float3 reflected = reflect(eye, normal);
    float m = 2.8284271247461903 * sqrt( reflected.z+1.0 );
    return reflected.xy / m + 0.5;
}
#endif