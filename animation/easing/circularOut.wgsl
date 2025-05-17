/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Circular out easing. From https://github.com/stackgl/glsl-easings
use: circularOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn circularOut(t: f32) -> f32 { return sqrt((2.0 - t) * t); }
