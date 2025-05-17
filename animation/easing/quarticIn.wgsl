/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quartic in easing. From https://github.com/stackgl/glsl-easings
use: quarticIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quarticIn(t: f32) -> f32 { return pow(t, 4.0); }
