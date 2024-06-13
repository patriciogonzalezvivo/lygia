/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Exponential out easing. From https://github.com/stackgl/glsl-easings
use: exponentialOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_EXPONENTIALOUT
#define FNC_EXPONENTIALOUT
float exponentialOut(in float t) { return t == 1.0 ? t : 1.0 - pow(2.0, -10.0 * t); }
#endif