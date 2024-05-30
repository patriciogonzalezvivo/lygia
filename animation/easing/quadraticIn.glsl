/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quadrtic in easing. From https://github.com/stackgl/glsl-easings
use: quadraticIn(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUADRATICIN
#define FNC_QUADRATICIN
float quadraticIn(in float t) { return t * t; }
#endif
