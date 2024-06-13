/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quartic in/out easing. From https://github.com/stackgl/glsl-easings
use: quarticInOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUARTICINOUT
#define FNC_QUARTICINOUT
float quarticInOut(in float t) {
    return t < 0.5
      ? +8.0 * pow(t, 4.0)
      : -8.0 * pow(t - 1.0, 4.0) + 1.0;
}
#endif