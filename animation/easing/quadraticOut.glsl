/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quadrtic out easing. From https://github.com/stackgl/glsl-easings
use: quadraticOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUADRATICOUT
#define FNC_QUADRATICOUT
float quadraticOut(in float t) { return -t * (t - 2.0); }
#endif