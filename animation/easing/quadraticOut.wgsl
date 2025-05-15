/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quadrtic out easing. From https://github.com/stackgl/glsl-easings
use: quadraticOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn quadraticOut(t: f32) -> f32 { return -t * (t - 2.0); }
