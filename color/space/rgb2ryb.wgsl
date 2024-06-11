#include "../../math/mmin.wgsl"
#include "../../math/mmax.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a color from RGB to RYB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <vec3f> ryb2rgb(<vec3f> ryb)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2ryb(_rgb: vec3f) -> vec3f {
    // Remove the white from the color
    let w = min3(_rgb);
    let bl = mmin3(1.0 - rgb);
    let rgb = _rgb - w;
        
    let max_g = mmax3(rgb);

    // Get the yellow out of the red & green
    let y = mmin2(rgb.rg);
    var ryb = rgb - vec3f(y, y, 0.);

    // If this unfortunate conversion combines blue and green, then cut each in half to preserve the value's maximum range.
    if (ryb.b > 0. && ryb.y > 0.) {
        ryb.b *= .5;
        ryb.y *= .5;
    }

    // Redistribute the remaining green.
    ryb.b += ryb.y;
    ryb.y += y;

    // Normalize to values.
    let max_y = mmax3(ryb);
    ryb *= (max_y > 0.) ? max_g / max_y : 1.;

    // Add the white back in.
    return ryb + bl;
}