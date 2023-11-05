/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Linear easing. From https://github.com/stackgl/glsl-easings
use: linear(<float> x)
*/

#ifndef FNC_LINEAR
#define FNC_LINEAR
float linearIn(in float t) {
    return t;
}

float linearOut(in float t) {
    return t;
}

float linearInOut(in float t) {
    return t;
}
#endif
