#include "stroke.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: create a bridge on a given in_value and draw a stroke inside that gap
use: bridge(<float|float2|float3|float4> in_value, <float> sdf, <float> size, <float> width)
*/

#ifndef FNC_BRIDGE
#define FNC_BRIDGE
float bridge(float c, float d, float s, float w) {
    c *= 1.0 - stroke(d, s , w * 2.0);
    return c + stroke(d, s, w);
}

float2 bridge(float2 c, float d, float s, float w) {
    c *= 1.0 - stroke(d, s , w * 2.0);
    return c + stroke(d, s, w);
}

float3 bridge(float3 c, float d, float s, float w) {
    c *= 1.0 - stroke(d, s , w * 2.0);
    return c + stroke(d, s, w);
}

float4 bridge(float4 c, float d, float s, float w) {
    c *= 1.0 - stroke(d, s , w * 2.0);
    return c + stroke(d, s, w);
}

#endif