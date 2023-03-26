/*
original_author: Hugh Kennedy (https://github.com/hughsk)
description: cubic easing. From https://github.com/stackgl/glsl-easings
use: cubic<In|Out|InOut>(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_CUBICIN
#define FNC_CUBICIN
float cubicIn(in float t) {
  return t * t * t;
}
#endif

#ifndef FNC_CUBICOUT
#define FNC_CUBICOUT
float cubicOut(in float t) {
  float f = t - 1.0;
  return f * f * f + 1.0;
}
#endif

#ifndef FNC_CUBICINOUT
#define FNC_CUBICINOUT
float cubicInOut(in float t) {
  return t < 0.5
    ? 4.0 * t * t * t
    : 0.5 * pow(2.0 * t - 2.0, 3.0) + 1.0;
}
#endif
