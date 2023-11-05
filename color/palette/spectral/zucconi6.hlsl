#include "../../../math/bump.hlsl"

/*
contributors: Alan Zucconi
description: |
    Read "Improving the Rainbow" for more information http://www.alanzucconi.com/?p=6703
    This provides the best approximation without including any branching.
    Based on GPU Gems: https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems/gpugems_ch08.html
    But with values optimised to match as close as possible the visible spectrum
    Fits this: https://commons.wikimedia.org/wiki/File:Linear_visible_spectrum.svg
    With weighter MSE (RGB weights: 0.3, 0.59, 0.11)
use: <float3> spectral_zucconi6(<float> x)
*/

#define SPECTRAL_ZUCCONI6_ITERATIONS 100.0

#ifndef FNC_SPECTRAL_ZUCCONI6
#define FNC_SPECTRAL_ZUCCONI6
float3 spectral_zucconi6 (float x) {
    const float3 c1 = float3(3.54585104, 2.93225262, 2.41593945);
    const float3 x1 = float3(0.69549072, 0.49228336, 0.27699880);
    const float3 y1 = float3(0.02312639, 0.15225084, 0.52607955);
    const float3 c2 = float3(3.90307140, 3.21182957, 3.96587128);
    const float3 x2 = float3(0.11748627, 0.86755042, 0.66077860);
    const float3 y2 = float3(0.84897130, 0.88445281, 0.73949448);
    return bump(c1 * (x - x1), y1) + bump(c2 * (x - x2), y2) ;
}
#endif