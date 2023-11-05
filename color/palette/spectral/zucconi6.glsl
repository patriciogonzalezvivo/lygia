#include "../../../math/bump.glsl"

/*
contributors: Alan Zucconi
description: |
    Read "Improving the Rainbow" for more information http://www.alanzucconi.com/?p=6703
    This provides the best approximation without including any branching.
    Based on GPU Gems: https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems/gpugems_ch08.html
    But with values optimised to match as close as possible the visible spectrum
    Fits this: https://commons.wikimedia.org/wiki/File:Linear_visible_spectrum.svg
    With weighter MSE (RGB weights: 0.3, 0.59, 0.11)
use: <vec3> spectral_zucconi6(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
*/

#define SPECTRAL_ZUCCONI6_ITERATIONS 100.0

#ifndef FNC_SPECTRAL_ZUCCONI6
#define FNC_SPECTRAL_ZUCCONI6
vec3 spectral_zucconi6 (float x) {
    const vec3 c1 = vec3(3.54585104, 2.93225262, 2.41593945);
    const vec3 x1 = vec3(0.69549072, 0.49228336, 0.27699880);
    const vec3 y1 = vec3(0.02312639, 0.15225084, 0.52607955);
    const vec3 c2 = vec3(3.90307140, 3.21182957, 3.96587128);
    const vec3 x2 = vec3(0.11748627, 0.86755042, 0.66077860);
    const vec3 y2 = vec3(0.84897130, 0.88445281, 0.73949448);
    return bump(c1 * (x - x1), y1) + bump(c2 * (x - x2), y2) ;
}
#endif