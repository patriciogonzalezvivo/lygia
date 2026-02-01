/*
contributors: ["Jos Stam", "Alias Systems"]
description: |
    From Chap 8 Simulating Diffraction from GPU Gems https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-8-simulating-diffraction
use: <vec3f> spectral_gems(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
*/

import lygia::math::bump::bumpDefault3;

fn spectral_gems(x: f32) -> vec3f {
    return bumpDefault3(vec3f(
        4.0 * (x - 0.75),   // Red
        4.0 * (x - 0.5),    // Green
        4.0 * (x - 0.25)    // Blue
    ));
}
