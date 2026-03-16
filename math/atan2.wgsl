#include "const.wgsl"

/*
contributors: Alexander Griffis
description: |
    The range here in degrees is 0 to pi (0-180 degrees) and -pi to 0 (181 to 359 degrees
    More about it at https://github.com/Yaazarai/GLSL-ATAN2-DOC

use: <float> atan2(<float>y, <float> x)
*/

fn atan2(y: f32, x: f32) -> f32 { return mod(atan(y,x) + PI, TAU); }
