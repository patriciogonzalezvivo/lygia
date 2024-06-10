#include "../../math/mmin.glsl"
#include "../../math/mmax.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from RYB to RGB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <vec3|vec4> ryb2rgb(<vec3|vec4> ryb)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RYB2RGB
#define FNC_RYB2RGB

vec3 ryb2rgb(vec3 ryb) {
    // Remove the white from the color
    float w = mmin(ryb);
    ryb -= w;

    float max_y = mmax(ryb);
        
    // Get the green out of the yellow & blue
    float g = min(ryb.g, ryb.b);
    vec3 rgb = ryb - vec3(0., g, g);
        
    if (rgb.b > 0. && g > 0.) {
        rgb.b   *= 2.;
        g   *= 2.;
    }

    // Redistribute the remaining yellow.
    rgb.r += rgb.g;
    rgb.g += g;

    // Normalize to values.
    float max_g = mmax(rgb);
    rgb *= (max_g > 0.) ? max_y / max_g : 1.;

    // Add the white back in.        
    return rgb + w;
}

vec4 ryb2rgb(vec4 ryb) { return vec4(ryb2rgb(ryb.rgb), ryb.a); }

#endif