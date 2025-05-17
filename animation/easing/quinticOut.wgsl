/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quintic out easing. From https://github.com/stackgl/glsl-easings
use: quinticOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quinticOut(t: f32) -> f32 { return 1.0 - (pow(t - 1.0, 5.0)); }
