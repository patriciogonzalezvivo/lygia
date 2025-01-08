#include "../../math/mmin.glsl"
#include "../../math/mmax.glsl"
#include "../../math/cubicMix.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a color from RGB to RYB color space. 
    Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf 
    and https://bahamas10.github.io/ryb/assets/ryb.pdf
use: <vec3|vec4> ryb2rgb(<vec3|vec4> ryb)
options:
    - RYB_HOMOGENEOUS: Use a non-homogeneous version of the conversion. Default is the homogeneous version.
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RYB_LERP 
#define RYB_LERP(A, B, t) cubicMix(A, B, t)
#endif

#ifndef FNC_RGB2RYB
#define FNC_RGB2RYB

#ifdef RYB_FAST

vec3 rgb2ryb(vec3 rgb) {
    // Remove the white from the color
    float w = mmin(rgb);
    float bl = mmin(1.0 - rgb);
    rgb -= w;
        
    float max_g = mmax(rgb);

    // Get the yellow out of the red & green
    float y = mmin(rgb.rg);
    vec3 ryb = rgb - vec3(y, y, 0.);

    // If this unfortunate conversion combines blue and green, then cut each in half to preserve the value's maximum range.
    if (ryb.b > 0. && ryb.y > 0.) {
        ryb.b *= .5;
        ryb.y *= .5;
    }

    // Redistribute the remaining green.
    ryb.b += ryb.y;
    ryb.y += y;

    // Normalize to values.
    float max_y = mmax(ryb);
    ryb *= (max_y > 0.) ? max_g / max_y : 1.;

    // Add the white back in.
#ifdef RYB_FAST
    return ryb + w;
#else
    return ryb + bl;
#endif
}

#else

vec3 rgb2ryb(vec3 rgb) {
    const vec3 rgb000 = vec3(1., 1., 1.);       // Black
    const vec3 rgb100 = vec3(1., 0., 0.);       // Red          
    const vec3 rgb010 = vec3(0., 1., .483);     // Green
    const vec3 rgb110 = vec3(0., 1., 0.);       // Yellow
    const vec3 rgb001 = vec3(0., 0., 1.);       // Blue
    const vec3 rgb101 = vec3(.309, 0., .469);   // Magenta
    const vec3 rgb011 = vec3(0., .053, .210);   // Turquoise
    const vec3 rgb111 = vec3(0., 0., 0.);       // White
    return RYB_LERP(RYB_LERP(
        RYB_LERP(rgb000, rgb001, rgb.z),
        RYB_LERP(rgb010, rgb011, rgb.z),
        rgb.y), RYB_LERP(
        RYB_LERP(rgb100, rgb101, rgb.z),
        RYB_LERP(rgb110, rgb111, rgb.z),
        rgb.y), rgb.x);
}

#endif

vec4 rgb2ryb(vec4 rgb) { return vec4(rgb2ryb(rgb.rgb), rgb.a); }

#endif
