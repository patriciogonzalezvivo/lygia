#include "../../../math/bump.glsl"

/*
contributors: ["Jos Stam", "Alias Systems"]
description: |
    From Chap 8 Simulating Diffraction from GPU Gems https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-8-simulating-diffraction
use: <vec3> spectral_gems(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
*/

#ifndef FNC_SPECTRAL_GEMS
#define FNC_SPECTRAL_GEMS
vec3 spectral_gems (float x) {
    return bump(vec3(   4. * (x - 0.75),    // Red
                        4. * (x - 0.5),     // Green
                        4. * (x - 0.25)     // Blue
                    ) );
}
#endif