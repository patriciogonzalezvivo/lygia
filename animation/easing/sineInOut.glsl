#include "../../math/const.glsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Sine in/out easing. From https://github.com/stackgl/glsl-easings
use: sineInOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_SINEINOUT
#define FNC_SINEINOUT
float sineInOut(in float t) { return -0.5 * (cos(PI * t) - 1.0); }
#endif
