/*
original_author: Hugh Kennedy (https://github.com/hughsk)
description: Linear easing. From https://github.com/stackgl/glsl-easings
use: linear(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_LINEAR
#define FNC_LINEAR
float linearIn(in float t) {
    return t;
}

float linearOut(in float t) {
    return t;
}

float linearInOUT(in float t) {
    return t;
}
#endif
