/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Cubic in/out easing. From https://github.com/stackgl/glsl-easings
use: cubicInOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_CUBICINOUT
#define FNC_CUBICINOUT
float cubicInOut(in float t) {
    return t < 0.5
      ? 4.0 * t * t * t
      : 0.5 * pow(2.0 * t - 2.0, 3.0) + 1.0;
}
#endif
