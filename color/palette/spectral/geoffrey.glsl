/*
contributors: none
description: none
use: <vec3> spectral_geoffrey(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
*/

#ifndef FNC_SPECTRAL_GEOFFREY
#define FNC_SPECTRAL_GEOFFREY
vec3 spectral_geoffrey(float t) {
    vec3 r = (t * 2.0 - 0.5) * 2.1 - vec3(1.8, 1.14, 0.3);
    return 0.99 - r * r;
}
#endif