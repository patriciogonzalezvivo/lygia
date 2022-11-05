/*
original_author: Patricio Gonzalez Vivo
description: Converts a Lab color to XYZ color space.
use: rgb2xyz(<vec3|vec4> color)
*/

#ifndef FNC_LAB2XYZ
#define FNC_LAB2XYZ
vec3 lab2xyz(in vec3 c) {
    vec3 f = vec3(0.0);
    f.y = (c.x + 16.0) / 116.0;
    f.x = c.y / 500.0 + f.y;
    f.z = f.y - c.z / 200.0;
    vec3 c0 = f * f * f;
    vec3 c1 = (f - 16.0 / 116.0) / 7.787;
    return vec3(95.047, 100.000, 108.883) * mix(c0, c1, step(f, vec3(0.206897)));
}

vec4 lab2xyz(in vec4 c) { return vec4(lab2xyz(c.xyz), c.w); }
#endif