#include "../../math/const.glsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Back in easing. From https://github.com/stackgl/glsl-easings
use: backIn(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_BACKIN
#define FNC_BACKIN
float backIn(in float t) { return pow(t, 3.) - t * sin(t * PI); }
#endif