/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quadrtic in/out easing. From https://github.com/stackgl/glsl-easings
use: quadraticInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quadraticInOut(t: f32) -> f32 {
  let p = 2.0 * t * t;
  return select(-p + (4.0 * t) - 1.0, p, t < 0.5);
}
