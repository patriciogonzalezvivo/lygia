/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lab color to XYZ color space.
    https://en.wikipedia.org/wiki/CIELAB_color_space
use: rgb2xyz(<vec3|vec4> color)
*/

#ifndef FNC_LAB2XYZ
#define FNC_LAB2XYZ
vec3 lab2xyz(const in vec3 c) {
    float fy = ( c.x + 16.0 ) / 116.0;
    float fx = c.y / 500.0 + fy;
    float fz = fy - c.z / 200.0;
    return vec3(
         95.047 * (( fx > 0.206897 ) ? fx * fx * fx : ( fx - 16.0 / 116.0 ) / 7.787),
        100.000 * (( fy > 0.206897 ) ? fy * fy * fy : ( fy - 16.0 / 116.0 ) / 7.787),
        108.883 * (( fz > 0.206897 ) ? fz * fz * fz : ( fz - 16.0 / 116.0 ) / 7.787)
    );
}

vec4 lab2xyz(in vec4 c) { return vec4(lab2xyz(c.xyz), c.w); }
#endif