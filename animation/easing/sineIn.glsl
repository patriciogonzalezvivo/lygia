#include "../../math/const.glsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Sine in easing. From https://github.com/stackgl/glsl-easings
use: sineIn(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_SINEIN
#define FNC_SINEIN
float sineIn(in float t) { return sin((t - 1.0) * HALF_PI) + 1.0; }
#endif