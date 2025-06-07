#include "bounceOut.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Bounce in/out easing. From https://github.com/stackgl/glsl-easings
use: <f32> bounceInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/animation/e_EasingBounce.frag
*/

fn bounceInOut(t: f32) -> f32 {
  return select(
    0.5 * bounceOut(t * 2.0 - 1.0) + 0.5,
    0.5 * (1.0 - bounceOut(1.0 - t * 2.0)),
    t < 0.5
  );
}
