#include "const.wgsl"

/*
contributors: Inigo Quiles
description: A band-limited variant of cos(x) which reduces aliasing at high frequencies. From https://iquilezles.org/articles/bandlimiting/
use: fcos(<float> value)
*/

fn fcos(x: f32) -> f32 {
    let w = fwidth(x);
    return cos(x) * smoothstep( TWO_PI, 0.0, w );
}
