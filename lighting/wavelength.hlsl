#include "../color/space/w2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Wavelength to RGB
use: <float3> wavelength(<float> wavelength)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WAVELENGTH
#define FNC_WAVELENGTH
float3 wavelength(float w) { return w2rgb(w); }
#endif
