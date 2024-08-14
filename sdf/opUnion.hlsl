#include "../lighting/material.hlsl"

/*
contributors:  Inigo Quiles
description: Union operation of two SDFs 
use: <float> opUnion( in <float|float4> d1, in <float|float4> d2 [, <float> smooth_factor] ) 
*/

#ifndef FNC_OPUNION
#define FNC_OPUNION

float opUnion( float d1, float d2 ) { return min(d1, d2); }

Material opUnion( Material d1, Material d2 ) {
    if (d1.sdf < d2.sdf) {
        return d1;
    } else {
        return d2;
    }
}

// Soft union
float opUnion( float d1, float d2, float k ) {
    float h = saturate( 0.5 + 0.5*(d2-d1)/k );
    return lerp( d2, d1, h ) - k*h*(1.0-h); 
}

float4 opUnion( float4 d1, float4 d2, float k ) {
    float h = saturate( 0.5 + 0.5*(d2.a - d1.a)/k );
    float4 result = lerp( d2, d1, h ); 
    result.a -= k*h*(1.0-h);
    return result;
}

#endif
