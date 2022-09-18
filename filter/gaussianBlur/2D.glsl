/*
original_author: Patricio Gonzalez Vivo
description: two dimension Gaussian Blur to be applied in only one passes
use: gaussianBlur2D(<sampler2D> texture, <vec2> st, <vec2> pixel_direction , const int kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GAUSSIANBLUR2D_TYPE: Default `vec4`
    - GAUSSIANBLUR2D_SAMPLER_FNC(POS_UV): Default `texture2D(tex, POS_UV)`
    - GAUSSIANBLUR2D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef GAUSSIANBLUR2D_TYPE
#ifdef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR2D_TYPE GAUSSIANBLUR_TYPE
#else
#define GAUSSIANBLUR2D_TYPE vec4
#endif
#endif

#ifndef GAUSSIANBLUR2D_SAMPLER_FNC
#ifdef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR2D_SAMPLER_FNC(POS_UV) GAUSSIANBLUR_SAMPLER_FNC(POS_UV)
#else
#define GAUSSIANBLUR2D_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif
#endif

#ifndef FNC_GAUSSIANBLUR2D
#define FNC_GAUSSIANBLUR2D
GAUSSIANBLUR2D_TYPE gaussianBlur2D(in sampler2D tex, in vec2 st, in vec2 offset, const int kernelSize) {
    GAUSSIANBLUR2D_TYPE accumColor = GAUSSIANBLUR2D_TYPE(0.);
    #ifndef GAUSSIANBLUR2D_KERNELSIZE
    #define GAUSSIANBLUR2D_KERNELSIZE kernelSize
    #endif

    float accumWeight = 0.;
    const float k = .15915494; // 1 / (2*PI)
    float kernelSize2 = float(GAUSSIANBLUR2D_KERNELSIZE) * float(GAUSSIANBLUR2D_KERNELSIZE);

    for (int j = 0; j < GAUSSIANBLUR2D_KERNELSIZE; j++) {
        float y = -.5 * (float(GAUSSIANBLUR2D_KERNELSIZE) - 1.) + float(j);
        for (int i = 0; i < GAUSSIANBLUR2D_KERNELSIZE; i++) {
            float x = -.5 * (float(GAUSSIANBLUR2D_KERNELSIZE) - 1.) + float(i);
            float weight = (k / float(GAUSSIANBLUR2D_KERNELSIZE)) * exp(-(x * x + y * y) / (2. * kernelSize2));
            accumColor += weight * GAUSSIANBLUR2D_SAMPLER_FNC(st + vec2(x, y) * offset);
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
#endif
