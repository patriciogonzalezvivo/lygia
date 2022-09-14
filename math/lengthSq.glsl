/*
original_author: Patricio Gonzalez Vivo
description: Squared length
use: lengthSq(<vec2|float2> st)
*/

#ifndef FNC_LENGTHSQ
#define FNC_LENGTHSQ
float lengthSq(in vec2 st) {
    return dot(st, st);
}

float lengthSq(in vec3 pos) {
    return dot(pos, pos);
}
#endif
