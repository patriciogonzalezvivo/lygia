/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Exponential in/out easing. From https://github.com/stackgl/glsl-easings
use: exponentialInOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_EXPONENTIALINOUT
#define FNC_EXPONENTIALINOUT
float exponentialInOut(in float t) {
    return t == 0.0 || t == 1.0
        ? t
        : t < 0.5
            ? +0.5 * pow(2.0, (20.0 * t) - 10.0)
            : -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
}
#endif