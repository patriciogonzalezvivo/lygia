/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Circular out easing. From https://github.com/stackgl/glsl-easings
use: circularOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_CIRCULAROUT
#define FNC_CIRCULAROUT
float circularOut(in float t) { return sqrt((2.0 - t) * t); }
#endif