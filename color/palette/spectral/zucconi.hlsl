#include "../../../math/bump.hlsl"

/*
contributors: Alan Zucconi
description: |
    Spectral Colour Schemes. Convert visible wavelengths of light (400-700 nm) to RGB colours http://www.alanzucconi.com/?p=6703
    Its faster version than spectral_zucconi6 advised for mobile applications.
    Read "Improving the Rainbow" for more information http://www.alanzucconi.com/?p=6703
    Based on GPU Gems: https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems/gpugems_ch08.html
    But with values optimised to match as close as possible the visible spectrum
    Fits this: https://commons.wikimedia.org/wiki/File:Linear_visible_spectrum.svg
    With weighter MSE (RGB weights: 0.3, 0.59, 0.11)
use: <float3> spectral_zucconi(<float> x)
*/

#ifndef FNC_SPECTRAL_ZUCCONI
#define FNC_SPECTRAL_ZUCCONI
float3 spectral_zucconi(float x) {
    const float3 cs = float3(3.54541723, 2.86670055, 2.29421995);
    const float3 xs = float3(0.69548916, 0.49416934, 0.28269708);
    const float3 ys = float3(0.02320775, 0.15936245, 0.53520021);
    return bump ( cs * (x - xs), ys);
}
#endif