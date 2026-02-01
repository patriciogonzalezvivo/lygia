/*
contributors: none
description: attempt attempt spectrum approximation
use: <vec3f> spectral_geoffrey(<f32> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
*/

fn spectral_geoffrey(t: f32) -> vec3f {
    let r = (t * 2.0 - 0.5) * 2.1 - vec3f(1.8, 1.14, 0.3);
    return 0.99 - r * r;
}
