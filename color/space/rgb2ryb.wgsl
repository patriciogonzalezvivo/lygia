/*
contributors: Patricio Gonzalez Vivo
description: Converts a color from RGB to RYB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <vec3> ryb2rgb(<vec3> ryb)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2ryb(_rgb: vec3f) -> vec3f {
    // Remove the white from the color
    let w = min(_rgb.r, min(_rgb.g, _rgb.b));
    let rgb = _rgb - w;
        
    let max_g = max(rgb.r, max(rgb.g, rgb.b));

    // Get the yellow out of the red & green
    let y = min(rgb.r, rgb.g);
    var ryb = rgb - vec3(y, y, 0.);

    // If this unfortunate conversion combines blue and green, then cut each in half to preserve the value's maximum range.
    if (ryb.b > 0. && ryb.y > 0.) {
        ryb.b *= .5;
        ryb.y *= .5;
    }

    // Redistribute the remaining green.
    ryb.b += ryb.y;
    ryb.y += y;

    // Normalize to values.
    let max_y = max(ryb.x, max(ryb.y, ryb.z);
    ryb *= (max_y > 0.) ? max_g / max_y : 1.;

    // Add the white back in.
    return ryb + w;
}