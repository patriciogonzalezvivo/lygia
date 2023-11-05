#include "../math/const.hlsl"

/*
contributors: Inigo Quiles
description:  Procedural generation of color palette algorithm explained here http://www.iquilezles.org/www/articles/palettes/palettes.htm)
use: palette(<float> t, <float3|float4> a, <float3|float4> b, <float3|float4> c, <float3|float4> d)
*/

#ifndef FNC_PALETTE
#define FNC_PALETTE
float3 palette (in float t, in float3 a, in float3 b, in float3 c, in float3 d) {
    return a + b * cos(TAU * ( c * t + d ));
}

float4 palette (in float t, in float4 a, in float4 b, in float4 c, in float4 d) {
    return a + b * cos(TAU * ( c * t + d ));
}
#endif
