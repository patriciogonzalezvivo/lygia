#include "rgb2srgb.glsl"
#include "srgb2oklab.glsl"

/*
contributors: Bjorn Ottosson (@bjornornorn)
description: |
    Linear rgb ot OKLab https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> rgb2oklab(<vec3|vec4> srgb)
*/

#ifndef FNC_RGB2OKLAB
#define FNC_RGB2OKLAB
vec3 rgb2oklab(const in vec3 rgb) { srgb2oklab( rgb2srgb(rgb) ); }
vec4 rgb2oklab(const in vec4 rgb) { return vec4(rgb2oklab(rgb.rgb), rgb.a); }
#endif