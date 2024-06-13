#include "srgb2rgb.glsl"
#include "rgb2oklab.glsl"

/*
contributors: Bjorn Ottosson (@bjornornorn)
description: |
    sRGB to OKLab https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> srgb2oklab(<vec3|vec4> srgb)
license: 
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
*/

#ifndef FNC_SRGB2OKLAB
#define FNC_SRGB2OKLAB
vec3 srgb2oklab(const in vec3 srgb) { return rgb2oklab( srgb2rgb(srgb) ); }
vec4 srgb2oklab(const in vec4 srgb) { return vec4(srgb2oklab(srgb.rgb), srgb.a); }
#endif
