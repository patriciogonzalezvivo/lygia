/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Cubic out easing. From https://github.com/stackgl/glsl-easings
use: cubicOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn cubicOut(t: f32) -> f32 {
    let f = t - 1.0;
    return f * f * f + 1.0;
}
