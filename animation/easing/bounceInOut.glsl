#include "bounceOut.glsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Bounce in/out easing. From https://github.com/stackgl/glsl-easings
use: <float> bounceInOut(<float> x)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/animation/e_EasingBounce.frag
*/

#ifndef FNC_BOUNCEINOUT
#define FNC_BOUNCEINOUT
float bounceInOut(in float t) {
    return t < 0.5
        ? 0.5 * (1.0 - bounceOut(1.0 - t * 2.0))
        : 0.5 * bounceOut(t * 2.0 - 1.0) + 0.5;
}
#endif