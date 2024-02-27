/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quadrtic in/out easing. From https://github.com/stackgl/glsl-easings
use: quadraticInOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUADRATICINOUT
#define FNC_QUADRATICINOUT
float quadraticInOut(in float t) {
    float p = 2.0 * t * t;
    return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
}
#endif
