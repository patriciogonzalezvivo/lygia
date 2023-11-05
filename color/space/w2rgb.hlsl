#include "gamma2linear.hlsl"
#include "../../color/palette/spectral.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: wavelength to RGB
use: <float3> w2rgb(<float> wavelength)
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