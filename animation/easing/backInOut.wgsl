#include "backIn.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Back in/out easing. From https://github.com/stackgl/glsl-easings
use: backInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn backInOut(t: f32) -> f32 {
  let g = backIn(select(1.0 - (2.0 * t - 1.0), 2.0 * t, t < 0.5));
  return select(0.5 * (1.0 - g) + 0.5, 0.5 * g, t < 0.5);
}
