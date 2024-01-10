#include "oklab2srgb.glsl"
#include "srgb2rgb.glsl"

/*
contributors: Bjorn Ottosson (@bjornornorn)
description: oklab to linear RGB https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> oklab2rgb(<vec3|vec4> oklab)
*/

#ifndef FNC_OKLAB2RGB
#define FNC_OKLAB2RGB
vec3 oklab2rgb(const in vec3 oklab) { return srgb2rgb(oklab2srgb(oklab)); }
vec4 oklab2rgb(const in vec4 oklab) { return vec4(oklab2rgb(oklab.xyz), oklab.a); }
#endif