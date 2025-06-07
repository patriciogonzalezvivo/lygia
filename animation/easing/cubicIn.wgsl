/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Cubic in easing. From https://github.com/stackgl/glsl-easings
use: cubicIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn cubicIn(t: f32) -> f32 { return t * t * t; }
