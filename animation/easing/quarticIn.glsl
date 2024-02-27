/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quartic in easing. From https://github.com/stackgl/glsl-easings
use: quarticIn(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUARTICIN
#define FNC_QUARTICIN
float quarticIn(in float t) { return pow(t, 4.0); }
#endif