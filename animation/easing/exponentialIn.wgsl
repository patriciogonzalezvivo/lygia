/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Exponential in easing. From https://github.com/stackgl/glsl-easings
use: exponentialIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn exponentialIn(t: f32) -> f32 {
  return select(
    pow(2.0, 10.0 * (t - 1.0)),
    t,
    t == 0.0
  );
}
