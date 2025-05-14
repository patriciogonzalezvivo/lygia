#include "bounceOut.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Bounce in easing. From https://github.com/stackgl/glsl-easings
use: <f32> bounceIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/animation/e_EasingBounce.frag
*/

fn bounceIn(t: f32) -> f32 { return 1.0 - bounceOut(1.0 - t); }
