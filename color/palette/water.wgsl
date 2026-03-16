#include "../../math/saturate.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simpler water color ramp
use: <vec3> water(<float> value)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/water.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn water(x: f32) -> vec3f {
    return pow(vec3f(.1, .7, .8), vec3f(4.* saturate(1.0-x) ));
}
