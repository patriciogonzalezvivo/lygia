#include "../../math/const.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Sine out easing. From https://github.com/stackgl/glsl-easings
use: sineOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn sineOut(t: f32) -> f32 { return sin(t * HALF_PI); }
