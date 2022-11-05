#include "linear2gamma.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a XYZ color to RGB color space.
use: xyz2rgb(<vec3|vec4> color)
*/

#ifndef FNC_XYZ2RGB
#define FNC_XYZ2RGB
vec3 xyz2rgb(in vec3 c) {
    const mat3 mat = mat3( 3.2404542, -0.9692660,  0.0556434,
                            -1.5371585,  1.8760108, -0.2040259,
                            -0.4985314,  0.0415560,  1.0572252);

    vec3 v = mat * (c / 100.0);
    vec3 c0 = (1.055 * linear2gamma(v)) - 0.055;
    vec3 c1 = 12.92 * v;
    vec3 r = mix(c0, c1, step(v, vec3(0.0031308)));
    return r;
}

vec4 xyz2rgb(in vec4 xyz) { return vec4(xyz2rgb(xyz.rgb), xyz.a); }
#endif