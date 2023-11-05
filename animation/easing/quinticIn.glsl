/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: quintic in easing. From https://github.com/stackgl/glsl-easings
use: quinticIn(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUINTICIN
#define FNC_QUINTICIN
float quinticIn(in float t) { return pow(t, 5.0); }
#endif
