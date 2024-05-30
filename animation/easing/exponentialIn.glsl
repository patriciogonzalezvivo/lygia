/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Exponential in easing. From https://github.com/stackgl/glsl-easings
use: exponentialIn(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_EXPONENTIALIN
#define FNC_EXPONENTIALIN
float exponentialIn(in float t) { return t == 0.0 ? t : pow(2.0, 10.0 * (t - 1.0)); }
#endif