#include "backIn.wgsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Back out easing. From https://github.com/stackgl/glsl-easings
use: backOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

fn backOut(t: f32) -> f32 { return 1.0 - backIn(1.0 - t); }
