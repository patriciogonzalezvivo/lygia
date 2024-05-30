/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Cubic out easing. From https://github.com/stackgl/glsl-easings
use: cubicOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_CUBICOUT
#define FNC_CUBICOUT
float cubicOut(in float t) {
    float f = t - 1.0;
    return f * f * f + 1.0;
}
#endif

