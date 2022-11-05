#include "gamma2linear.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a RGB color to XYZ color space.
use: rgb2xyz(<vec3|vec4> color)
*/

#ifndef FNC_RGB2XYZ
#define FNC_RGB2XYZ
vec3 rgb2xyz(in vec3 c) {
    const mat3 mat = mat3(  0.4124564, 0.2126729, 0.0193339,
                            0.3575761, 0.7151522, 0.1191920,
                            0.1804375, 0.0721750, 0.9503041);
    vec3 c0 = gamma2linear((c + 0.055) / 1.055);
    vec3 c1 = c / 12.92;
    vec3 tmp = mix(c0, c1, step(c, vec3(0.04045)));
    return mat * (100.0 * tmp);
}

vec4 rgb2xyz(in vec4 rgb) { return vec4(rgb2xyz(rgb.rgb),rgb.a); }
#endif