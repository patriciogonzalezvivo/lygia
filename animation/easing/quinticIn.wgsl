/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: quintic in easing. From https://github.com/stackgl/glsl-easings
use: quinticIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quinticIn(t: f32) -> f32 { return pow(t, 5.0); }
