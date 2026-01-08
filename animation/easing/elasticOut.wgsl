#include "../../math/const.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Elastic out easing. From https://github.com/stackgl/glsl-easings
use: elasticOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn elasticOut(t: f32) -> f32 { return sin(-13.0 * (t + 1.0) * HALF_PI) * pow(2.0, -10.0 * t) + 1.0; }
