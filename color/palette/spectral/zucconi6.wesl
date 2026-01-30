/*
contributors: Alan Zucconi
description: |
    Read "Improving the Rainbow" for more information http://www.alanzucconi.com/?p=6703
    This provides the best approximation without including any branching.
    Based on GPU Gems: https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems/gpugems_ch08.html
    But with values optimised to match as close as possible the visible spectrum
    Fits this: https://commons.wikimedia.org/wiki/File:Linear_visible_spectrum.svg
    With weighter MSE (RGB weights: 0.3, 0.59, 0.11)
use: <vec3f> spectral_zucconi6(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
*/

import lygia::math::bump::bump3;

const SPECTRAL_ZUCCONI6_ITERATIONS: f32 = 100.0;

fn spectral_zucconi6(x: f32) -> vec3f {
    const c1 = vec3f(3.54585104, 2.93225262, 2.41593945);
    const x1 = vec3f(0.69549072, 0.49228336, 0.27699880);
    const y1 = vec3f(0.02312639, 0.15225084, 0.52607955);
    const c2 = vec3f(3.90307140, 3.21182957, 3.96587128);
    const x2 = vec3f(0.11748627, 0.86755042, 0.66077860);
    const y2 = vec3f(0.84897130, 0.88445281, 0.73949448);
    return bump3(c1 * (x - x1), y1) + bump3(c2 * (x - x2), y2);
}
