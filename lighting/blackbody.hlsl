#include "../color/space/k2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Blackbody in kelvin to RGB. Range between 0.0 and 40000.0 Kelvin
use: <float3> blackbody(<float> wavelength)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_BLACKBODY
#define FNC_BLACKBODY
float3 blackbody(const in float k) { return k2rgb(k); }
#endif