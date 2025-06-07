#include "../../math/const.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Back in easing. From https://github.com/stackgl/glsl-easings
use: backIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn backIn(t: f32) -> f32 { return pow(t, 3.0) - t * sin(t * PI); }
