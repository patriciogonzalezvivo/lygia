/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Exponential in/out easing. From https://github.com/stackgl/glsl-easings
use: exponentialInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn exponentialInOut(t: f32) -> f32 {
  return select(
    select(
      -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0,
      0.5 * pow(2.0, (20.0 * t) - 10.0),
      t < 0.5
    ),
    t,
    t == 0.0 || t == 1.0
  );
}
