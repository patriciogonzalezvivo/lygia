/*
contributors: Patricio Gonzalez Vivo
description: Convert from RYB to RGB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <vec3> ryb2rgb(<vec3> ryb)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn ryb2rgb(_ryb: vec3f) -> vec3f {
    // Remove the white from the color
    let w = min(_ryb.x, min(_ryb.y, _ryb.z);
    let ryb = _ryb - w;

    let max_y = max(ryb.x, max(ryb.y, ryb.z));
        
    // Get the green out of the yellow & blue
    var g = min(ryb.g, ryb.b);
    var rgb = ryb - vec3f(0., g, g);
        
    if (rgb.b > 0. && g > 0.) {
        rgb.b *= 2.;
        g *= 2.;
    }

    // Redistribute the remaining yellow.
    rgb.r += rgb.g;
    rgb.g += g;

    // Normalize to values.
    let max_g = max(rgb.x, max(rgb.y, rgb.z));
    rgb *= (max_g > 0.) ? max_y / max_g : 1.;

    // Add the white back in.        
    return rgb + w;
}
