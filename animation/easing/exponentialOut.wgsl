/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Exponential out easing. From https://github.com/stackgl/glsl-easings
use: exponentialOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn exponentialOut(t: f32) -> f32 {
  return select(
    1.0 - pow(2.0, -10.0 * t),
    t,
    t == 1.0
  );
}
