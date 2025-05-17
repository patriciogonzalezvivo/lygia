#include "../../math/const.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Sine in/out easing. From https://github.com/stackgl/glsl-easings
use: sineInOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn sineInOut(t: f32) -> f32 { return -0.5 * (cos(PI * t) - 1.0); }
