/*
original_author: [Ian McEwan, Ashima Arts]
description: grad4, used for snoise(float4 v)
use: grad4(<float> j, <float4> ip)

*/
#ifndef FNC_GRATERTHAN
#define FNC_GRATERTHAN

float greaterThan(float x, float y) {
    // return x >
    return step(y, x);
}

float2 greaterThan(float2 x, float2 y) {
    
    return step(y, x);
}

float3 greaterThan(float3 x, float3 y) {
    
    return step(y, x);
}

float4 greaterThan(float4 x, float4 y) {
    
    return step(y, x);
}
    
#endif
