#include "backIn.glsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Back in/out easing. From https://github.com/stackgl/glsl-easings
use: backInOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_BACKINOUT
#define FNC_BACKINOUT
float backInOut(in float t) {
    float f = t < .5
        ? 2.0 * t
        : 1.0 - (2.0 * t - 1.0);

    float g = backIn(f);

    return t < 0.5
        ? 0.5 * g
        : 0.5 * (1.0 - g) + 0.5;
}
#endif
