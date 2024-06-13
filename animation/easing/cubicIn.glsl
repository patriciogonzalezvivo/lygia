/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Cubic in easing. From https://github.com/stackgl/glsl-easings
use: cubicIn(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_CUBICIN
#define FNC_CUBICIN
float cubicIn(in float t) { return t * t * t; }
#endif