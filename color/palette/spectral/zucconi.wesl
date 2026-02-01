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
use: <vec3f> spectral_zucconi(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
*/

import lygia::math::bump::bump3;

fn spectral_zucconi(x: f32) -> vec3f {
    const cs = vec3f(3.54541723, 2.86670055, 2.29421995);
    const xs = vec3f(0.69548916, 0.49416934, 0.28269708);
    const ys = vec3f(0.02320775, 0.15936245, 0.53520021);
    return bump3(cs * (x - xs), ys);
}
