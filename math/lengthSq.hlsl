/*
original_author: Patricio Gonzalez Vivo
description: Squared length
use: lengthSq(<float2|float3> st)
*/

#ifndef FNC_LENGTHSQ
#define FNC_LENGTHSQ
float lengthSq(in float2 st) {
    return dot(st, st);
}

float lengthSq(in float3 pos) {
    return dot(pos, pos);
}
#endif
