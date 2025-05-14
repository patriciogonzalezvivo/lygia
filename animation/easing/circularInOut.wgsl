/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Circular in/out easing. From https://github.com/stackgl/glsl-easings
use: circularInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn circularInOut(t: f32) -> f32 {
  return select(
    0.5 * (sqrt((3.0 - 2.0 * t) * (2.0 * t - 1.0)) + 1.0),
    0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t)),
    t < 0.5
  );
}
