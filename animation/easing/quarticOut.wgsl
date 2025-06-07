/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quartic out easing. From https://github.com/stackgl/glsl-easings
use: quarticOut<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quarticOut(t: f32) -> f32 { 
  let it = t - 1.0;
  return it * it * it * (1.0 - t) + 1.0; 
}
