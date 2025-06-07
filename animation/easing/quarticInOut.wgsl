/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quartic in/out easing. From https://github.com/stackgl/glsl-easings
use: quarticInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quarticInOut(t: f32) -> f32 {
  return select(
    -8.0 * pow(t - 1.0, 4.0) + 1.0,
    8.0 * pow(t, 4.0),
    t < 0.5
  );
}
