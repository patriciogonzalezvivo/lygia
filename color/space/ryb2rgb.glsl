#include "../../math/mmin.glsl"
#include "../../math/mmax.glsl"
#include "../../math/cubicMix.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from RYB to RGB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf http://vis.computer.org/vis2004/DVD/infovis/papers/gossett.pdf
use: <vec3|vec4> ryb2rgb(<vec3|vec4> ryb)
options:
    - RYB_FAST: Use a non-homogeneous version of the conversion. Default is the homogeneous version.
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RYB_LERP 
#define RYB_LERP(A, B, t) cubicMix(A, B, t)
#endif

#ifndef FNC_RYB2RGB
#define FNC_RYB2RGB

#ifdef RYB_FAST

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
    const vec3 ryb000 = vec3(1., 1., 1.);       // white
    const vec3 ryb100 = vec3(1., 0., 0.);       // Red          
    const vec3 ryb010 = vec3(1., 1., 0.);       // Yellow
    const vec3 ryb110 = vec3(1., .5, 0.);       // Orange
    const vec3 ryb001 = vec3(.163, .373, .6);   // blue
    const vec3 ryb101 = vec3(.5, 0., .5);       // Purple
    const vec3 ryb011 = vec3(0., .66, .2);      // Green
    const vec3 ryb111 = vec3(0., 0., 0.);       // Black
    return RYB_LERP(RYB_LERP(
        RYB_LERP(ryb000, ryb001, ryb.z),
        RYB_LERP(ryb010, ryb011, ryb.z),
        ryb.y), RYB_LERP(
        RYB_LERP(ryb100, ryb101, ryb.z),
        RYB_LERP(ryb110, ryb111, ryb.z),
        ryb.y), ryb.x);
}

#endif

vec4 ryb2rgb(vec4 ryb) { return vec4(ryb2rgb(ryb.rgb), ryb.a); }

#endif