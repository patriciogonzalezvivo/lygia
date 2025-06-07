#include "../../math/const.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Elastic in/out easing. From https://github.com/stackgl/glsl-easings
use: elasticInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn elasticInOut(t: f32) -> f32 {
  return select(
    0.5 * sin(-13.0 * HALF_PI * ((2.0 * t - 1.0) + 1.0)) * pow(2.0, -10.0 * (2.0 * t - 1.0)) + 1.0,
    0.5 * sin(13.0 * HALF_PI * 2.0 * t) * pow(2.0, 10.0 * (2.0 * t - 1.0)),
    t < 0.5
  );
}
