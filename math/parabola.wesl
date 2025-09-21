/*
contributors: Inigo Quiles
description: |
    A nice choice to remap the 0..1 interval into 0..1, such that the corners are mapped to 0 and the center to 1. You can then rise the parabolar to a power k to control its shape. From https://iquilezles.org/articles/functions/
*/

fn parabola(x: f32, k: f32) -> f32 { return pow( 4.0*x*(1.0-x), k ); }
