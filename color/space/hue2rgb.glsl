#include "../../math/saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a hue value to a RGB vec3 color.'
use: <vec3> hue2rgb(<float> hue)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HUE2RGB
#define FNC_HUE2RGB
vec3 hue2rgb(const in float hue) {
    float R = abs(hue * 6.0 - 3.0) - 1.0;
    float G = 2.0 - abs(hue * 6.0 - 2.0);
    float B = 2.0 - abs(hue * 6.0 - 4.0);
    return saturate(vec3(R,G,B));
}
#endif