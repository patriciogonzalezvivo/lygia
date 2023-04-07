#include "rgb2srgb.glsl"

/*
original_author: Ronald van Wijnen (@OneDayOfCrypto)
description: Converts a XYZ color to sRGB color space.
use: xyz2rgb(<vec3|vec4> color)
*/

#ifndef FNC_XYZ2SRGB
#define FNC_XYZ2SRGB
vec3 xyz2srgb(vec3 xyz) {
    mat3 D65_XYZ_RGB;
    D65_XYZ_RGB[0] = vec3( 3.24306333, -1.53837619, -0.49893282);
    D65_XYZ_RGB[1] = vec3(-0.96896309,  1.87542451,  0.04154303);
    D65_XYZ_RGB[2] = vec3( 0.05568392, -0.20417438,  1.05799454);
    
    float r = dot(D65_XYZ_RGB[0], xyz);
    float g = dot(D65_XYZ_RGB[1], xyz);
    float b = dot(D65_XYZ_RGB[2], xyz);
    return rgb2srgb(vec3(r, g, b));
}

vec4 xyz2srgb(in vec4 xyz) { return vec4(xyz2srgb(xyz.rgb), xyz.a); }
#endif

