#include "bounceOut.glsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Bounce in easing. From https://github.com/stackgl/glsl-easings
use: <float> bounceIn(<float> x)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/animation/e_EasingBounce.frag
*/

#ifndef FNC_BOUNCEIN
#define FNC_BOUNCEIN
float bounceIn(in float t) { return 1.0 - bounceOut(1.0 - t); }
#endif
