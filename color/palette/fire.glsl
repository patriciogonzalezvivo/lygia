/*
contributors: Patricio Gonzalez Vivo
description: Simpler fire color ramp
use: <vec3> fire(<float> value)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/fire.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FIRE
#define FNC_FIRE
vec3 fire(float x) { return vec3(1.0, 0.25, 0.0625) * exp(4.0 * x - 1.0); }
#endif