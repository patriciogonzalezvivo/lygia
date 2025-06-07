/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quintic in/out easing. From https://github.com/stackgl/glsl-easings
use: quinticInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quinticInOut(t: f32) -> f32 {
  return select(
    -0.5 * pow(2.0 * t - 2.0, 5.0) + 1.0,
    16.0 * pow(t, 5.0),
    t < 0.5
  );
}
