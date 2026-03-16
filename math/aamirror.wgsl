#include "nyquist.wgsl"

/*
contributors: Shadi El Hajj
description: An anti-aliased triangle wave function.
use: <float> mirror(<float> x)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn aamirror(x: f32) -> f32 {
    let afwidth = AA_EDGE;
    let afwidth = length(vec2f(dpdx(x),dpdy(x)));
    let v = abs(x - floor(x + 0.5)) * 2.0;
    return nyquist(v, afwidth);
}
