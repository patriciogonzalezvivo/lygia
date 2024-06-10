#include "gamma2linear.hlsl"
#include "../../color/palette/spectral.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Wavelength to RGB
use: <float3> w2rgb(<float> wavelength)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_W2RGB
#define FNC_W2RGB
float3 w2rgb(float w) {

    #if defined(W2RGB_APPROXIMATION_FNC)
    float x = saturate((w - 400.0)/ 300.0);
    return gamma2linear( W2RGB_APPROXIMATION_FNC(x) );
    #else

    #endif
}

#endif