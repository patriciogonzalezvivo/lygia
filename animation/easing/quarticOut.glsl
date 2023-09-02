/*
original_author: Hugh Kennedy (https://github.com/hughsk)
description: quartic out easing. From https://github.com/stackgl/glsl-easings
use: quarticOut<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUARTICOUT
#define FNC_QUARTICOUT
float quarticOut(in float t) {
  return pow(t - 1.0, 3.0) * (1.0 - t) + 1.0;
}
#endif