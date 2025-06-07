/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Cubic in/out easing. From https://github.com/stackgl/glsl-easings
use: cubicInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn cubicInOut(t: f32) -> f32 {
  return select(
    0.5 * pow(2.0 * t - 2.0, 3.0) + 1.0,
    4.0 * t * t * t,
    t < 0.5
  );
}
