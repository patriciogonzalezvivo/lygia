#include "gamma2linear.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: wavelength to RGB
use: <float3> w2rgb(<float> wavelength)
*/

#ifndef FNC_W2RGB
#define FNC_W2RGB
float3 w2rgb(float w) {
    float x = saturate((w - 400.0)/ 300.0);
    const float3 c1 = float3(3.54585104, 2.93225262, 2.41593945);
    const float3 x1 = float3(0.69549072, 0.49228336, 0.27699880);
    const float3 y1 = float3(0.02312639, 0.15225084, 0.52607955);
    const float3 c2 = float3(3.90307140, 3.21182957, 3.96587128);
    const float3 x2 = float3(0.11748627, 0.86755042, 0.66077860);
    const float3 y2 = float3(0.84897130, 0.88445281, 0.73949448);
    return gamma2linear( bump(c1 * (x - x1), y1) + bump(c2 * (x - x2), y2) );
}
#endif