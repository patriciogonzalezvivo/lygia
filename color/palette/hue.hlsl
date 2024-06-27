#include "../../math/mod.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Physical Hue. Ratio: 1/3 = neon, 1/4 = refracted, 1/5+ = approximate white'
use: <float3> hue(<float> hue[, <float> ratio])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_PALETTE_HUE
#define FNC_PALETTE_HUE

float3 hue(float x, float r) {
    float3 v = abs(mod(frac(1.0 - x) + float3(0.0, 1.0, 2.0) * r, 1.0) * 2.0 - 1.0);
    return v * v * (3.0 - 2.0 * v);
}

#endif