/*
contributors: Patricio Gonzalez Vivo
description: "Physical Hue. \n\nRatio: \n* 1/3 = neon\n* 1/4 = refracted\n* 1/5+ =\ approximate white\n"
use: <vec3> hue(<float> hue[, <float> ratio])
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/hue.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_PALETTE_HUE
#define FNC_PALETTE_HUE
vec3 hue(float x, float r) { 
    vec3 v = abs( mod(fract(1.0-x) + vec3(0.0,1.0,2.0) * r, 1.0) * 2.0 - 1.0); 
    return v*v*(3.0-2.0*v);
}
vec3 hue(float x) { return hue(x, 0.33333); }
#endif