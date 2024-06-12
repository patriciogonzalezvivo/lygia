#include "../../sampler.glsl"

/*
contributors: Johan Ismael
description: Sharpening convolutional operation
use: sharpen(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SHARPENFAST_KERNELSIZE: Defaults 2
    - SHARPENFAST_TYPE: defaults to vec3
    - SHARPENFAST_SAMPLER_FNC(TEX, UV): defaults to texture2D(tex, TEX, UV).rgb
*/

#ifndef SHARPENFAST_KERNELSIZE
#ifdef SHARPEN_KERNELSIZE
#define SHARPENFAST_KERNELSIZE SHARPEN_KERNELSIZE
#else
#define SHARPENFAST_KERNELSIZE 2
#endif
#endif

#ifndef SHARPENFAST_TYPE
#ifdef SHARPEN_TYPE
#define SHARPENFAST_TYPE SHARPEN_TYPE
#else
#define SHARPENFAST_TYPE vec3
#endif
#endif

#ifndef SHARPENFAST_SAMPLER_FNC
#ifdef SHARPEN_SAMPLER_FNC
#define SHARPENFAST_SAMPLER_FNC(TEX, UV) SHARPEN_SAMPLER_FNC(TEX, UV)
#else
#define SHARPENFAST_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb
#endif
#endif

#ifndef FNC_SHARPENFAST
#define FNC_SHARPENFAST
SHARPENFAST_TYPE sharpenFast(in SAMPLER_TYPE tex, in vec2 coords, in vec2 pixel, float strength) {
    SHARPENFAST_TYPE sum = SHARPENFAST_TYPE(0.);
    for (int i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        float f_size = float(i) + 1.;
        f_size *= strength;
        sum += -1. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( -1., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( 0., -1.) * pixel * f_size);
        sum +=  5. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( 0., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( 0., 1.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( 1., 0.) * pixel * f_size);
    }
    return sum / float(SHARPENFAST_KERNELSIZE);
}

SHARPENFAST_TYPE sharpenFast(in SAMPLER_TYPE tex, in vec2 coords, in vec2 pixel) {
    SHARPENFAST_TYPE sum = SHARPENFAST_TYPE(0.);
    for (int i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        float f_size = float(i) + 1.;
        sum += -1. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( -1., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( 0., -1.) * pixel * f_size);
        sum +=  5. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( 0., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( 0., 1.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(tex, coords + vec2( 1., 0.) * pixel * f_size);
    }
    return sum / float(SHARPENFAST_KERNELSIZE);
}
#endif