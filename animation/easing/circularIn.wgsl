/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Circular in easing. From https://github.com/stackgl/glsl-easings
use: circularIn(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn circularIn(t: f32) -> f32 { return 1.0 - sqrt(1.0 - t * t); }
