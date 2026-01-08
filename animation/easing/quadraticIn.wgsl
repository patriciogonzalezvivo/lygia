/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quadrtic in easing. From https://github.com/stackgl/glsl-easings
use: quadraticIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quadraticIn(t: f32) -> f32 { return t * t; }
