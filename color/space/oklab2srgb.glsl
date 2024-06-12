#include "oklab2rgb.glsl"
#include "rgb2srgb.glsl"

/*
contributors: Bjorn Ottosson (@bjornornorn)
description: Oklab to sRGB https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> oklab2srgb(<vec3|vec4> oklab)
license: 
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
*/

#ifndef FNC_OKLAB2SRGB
#define FNC_OKLAB2SRGB
vec3 oklab2srgb(const in vec3 oklab) { return rgb2srgb(oklab2rgb(oklab)); }
vec4 oklab2srgb(const in vec4 oklab) { return vec4(oklab2srgb(oklab.xyz), oklab.a); }
#endif
