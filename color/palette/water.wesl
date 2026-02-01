/*
contributors: Patricio Gonzalez Vivo
description: Simpler water color ramp
use: <vec3f> water(<f32> value)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/water.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import lygia::math::saturate::saturate;

fn water(x: f32) -> vec3f {
    return pow(vec3f(0.1, 0.7, 0.8), vec3f(4.0 * saturate(1.0 - x)));
}
