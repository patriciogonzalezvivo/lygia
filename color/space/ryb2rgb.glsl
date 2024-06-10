#include "../../math/mmin.glsl"
#include "../../math/mmax.glsl"
#include "../../math/cubicMix.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from RYB to RGB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <vec3|vec4> ryb2rgb(<vec3|vec4> ryb)
options:
    - RYB_SMOOTH: Use a non-homogeneous version of the conversion. Default is the homogeneous version.
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RYB2RGB
#define FNC_RYB2RGB

#ifndef RYB_SMOOTH

vec3 ryb2rgb(vec3 ryb) {
    // Remove the white from the color
    float w = mmin(ryb);
    ryb -= w;

    float max_y = mmax(ryb);
        
    // Get the green out of the yellow & blue
    float g = mmin(ryb.gb);
    vec3 rgb = ryb - vec3(0., g, g);
        
    if (rgb.b > 0. && g > 0.) {
        rgb.b *= 2.;
        g *= 2.;
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

#else

vec3 ryb2rgb(vec3 ryb) {
    vec3 rgb = vec3(0.0, 0.0, 0.0);
    //red
    vec4 X = cubicMix(vec4(1.0), vec4(0.163, 0.0, 0.5, 0.2), ryb.z);
    vec2 Y = cubicMix(X.xz, X.yw, ryb.y);
    rgb.r  = cubicMix(Y.x, Y.y, ryb.x);
    //green
    X      = cubicMix(vec4(1.0, 1.0, 0.0, 0.5), vec4(0.373, 0.66, 0.0, 0.094), ryb.z);
    Y      = cubicMix(X.xz, X.yw, ryb.y);
    rgb.g  = cubicMix(Y.x, Y.y, ryb.x);
    //blue
    X      = cubicMix(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.6, 0.2, 0.5, 0.0), ryb.z);
    Y      = cubicMix(X.xz, X.yw, ryb.y);
    rgb.b  = cubicMix(Y.x, Y.y, ryb.x);
    return rgb;
}

#endif

vec4 ryb2rgb(vec4 ryb) { return vec4(ryb2rgb(ryb.rgb), ryb.a); }

#endif