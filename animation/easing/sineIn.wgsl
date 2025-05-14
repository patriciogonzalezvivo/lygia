#include "../../math/const.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Sine in easing. From https://github.com/stackgl/glsl-easings
use: sineIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn sineIn(t: f32) -> f32 { return sin((t - 1.0) * HALF_PI) + 1.0; }
