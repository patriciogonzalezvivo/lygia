#include "../../math/const.glsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Elastic out easing. From https://github.com/stackgl/glsl-easings
use: elasticOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_ELASTICOUT
#define FNC_ELASTICOUT
float elasticOut(in float t) { return sin(-13.0 * (t + 1.0) * HALF_PI) * pow(2.0, -10.0 * t) + 1.0; }
#endif
