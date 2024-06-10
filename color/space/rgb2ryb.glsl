#include "../../math/mmin.glsl"
#include "../../math/mmax.glsl"

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

#ifndef FNC_RGB2RYB
#define FNC_RGB2RYB

vec3 rgb2ryb(vec3 rgb) {
    // Remove the white from the color
    float w = mmin(rgb);
    rgb -= w;
        
    float max_g = mmax(rgb);

    // Get the yellow out of the red & green
    float y = min(rgb.r, rgb.g);
    vec3 ryb = rgb - vec3(y, y, 0.);

    // If this unfortunate conversion combines blue and green, then cut each in half to preserve the value's maximum range.
    if (ryb.b > 0. && ryb.y > 0.) {
        ryb.b   *= 0.5;
        ryb.y   *= 0.5;
    }

    // Redistribute the remaining green.
    ryb.b   += ryb.y;
    ryb.y   += y;

    // Normalize to values.
    float max_y = mmax(ryb);
    ryb *= (max_y > 0.) ? max_g / max_y : 1.;

    // Add the white back in.
    return ryb + w;
}

vec4 rgb2ryb(vec4 rgb) { return vec4(rgb2ryb(rgb.rgb), rgb.a); }

#endif
