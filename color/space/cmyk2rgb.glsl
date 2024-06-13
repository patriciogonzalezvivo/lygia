#include "../../math/saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert CMYK to RGB
use: cmyk2rgb(<vec4> cmyk)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_CMYK2RGB
#define FNC_CMYK2RGB
vec3 cmyk2rgb(const in vec4 cmyk) {
    float invK = 1.0 - cmyk.w;
    return saturate(1.0-min(vec3(1.0), cmyk.xyz * invK + cmyk.w));
}
#endif