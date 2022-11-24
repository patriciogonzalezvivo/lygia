#include "../../space/nearest.glsl"

#include "../../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: one dimensional bilateral Blur that use a blue noise texture to sample a kernel with out introducing biases
use: bilateralBlurFast(<sampler2D> texture, <vec2> st, <vec2> pixelSize, <float> smoothingFactor,  <float> kernelSize)
options:
    - BILATERALBLURFAST_TYPE: default is vec4
    - BILATERALBLURFAST_SAMPLER_FNC(TEX, UV): default texture2D(tex, TEX, UV)
    - BILATERALBLURFAST_NOISE_TEX_SIZE: blue noise texture size. Default 64.0
    - BILATERALBLURFAST_SIGMA_R: sigma defualt .075
    - BILATERALBLURFAST_NOISE_FNC: functions use to sample the blue noise texture
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

#ifndef BILATERALBLURFAST_TYPE
#ifdef BILATERALBLUR_TYPE
#define BILATERALBLURFAST_TYPE BILATERALBLUR_TYPE
#else
#define BILATERALBLURFAST_TYPE vec4
#endif
#endif

#ifndef BILATERALBLURFAST_SAMPLER_FNC
#ifdef BILATERALBLUR_SAMPLER_FNC
#define BILATERALBLURFAST_SAMPLER_FNC(TEX, UV) BILATERALBLUR_SAMPLER_FNC(TEX, UV)
#else
#define BILATERALBLURFAST_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef BILATERALBLURFAST_NOISE_TEX_SIZE
#define BILATERALBLURFAST_NOISE_TEX_SIZE 64.0
#endif

#ifndef BILATERALBLURFAST_SIGMA_R
#define BILATERALBLURFAST_SIGMA_R .075
#endif

#ifndef BILATERALBLURFAST_NOISE_FNC
#define BILATERALBLURFAST_NOISE_FNC(TEX, UV) SAMPLER_FNC(poissonNoise, nearest(fract(TEX, UV),vec2(BILATERALBLURFAST_NOISE_TEX_SIZE))).xy
#endif

#ifndef FNC_BILATERALBLURFAST
#define FNC_BILATERALBLURFAST

#if define(PLATFORM_RPI)
BILATERALBLURFAST_TYPE bilateralBlurFast(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor, const float sigma_s) {
    BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(tex, st);
    BILATERALBLURFAST_TYPE accumColor = BILATERALBLURFAST_TYPE(0.);
    float accumWeight = 0.;

    float scaleY2 = (BILATERALBLURFAST_NOISE_TEX_SIZE * pixel.y);
    float sigma_s2 = 1. / (2. * sigma_s * sigma_s);
    float kernelSize = ceil(5. * sigma_s);
    vec2 scale = kernelSize * pixel;

    float yFetch = st.y * scaleY2;
    float numSamples = 2. * kernelSize;
    float samplePitch = 1.0 / numSamples;

    float sigma_r = .01 * (1. - smoothingFactor) + .1 * smoothingFactor;
    float sigma_r2 = 1. / (2. * sigma_r * sigma_r);

    for (float i = 0.0; i < numSamples && accumWeight <= .25 * numSamples; i++) {
        vec2 coords = BILATERALBLURFAST_NOISE_FNC(vec2(i * samplePitch, yFetch));
        coords = (coords - .5) * scale;
        float coordsz = dot(coords, coords);
        BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(tex, coords + st);
        BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
        float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * sigma_s2);
        accumColor += colorFetch * tmpWeight;
        accumWeight += tmpWeight;
    }
    return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}
#endif

BILATERALBLURFAST_TYPE bilateralBlurFast10(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
    BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(tex, st);
    BILATERALBLURFAST_TYPE accumColor = BILATERALBLURFAST_TYPE(0.);
    float accumWeight = 0.;

    float scaleY2 = (BILATERALBLURFAST_NOISE_TEX_SIZE * pixel.y);
    vec2 scale = 5. * pixel;

    float yFetch = st.y * scaleY2;

    float sigma_r = .01 * (1. - smoothingFactor) + .1 * smoothingFactor;
    float sigma_r2 = 1. / (2. * sigma_r * sigma_r);

    for (float i = 0.0; i < 10. && accumWeight <= 2.5; i++) {
        vec2 coords = BILATERALBLURFAST_NOISE_FNC(vec2(i * .1, yFetch));
        coords = (coords - .5) * scale;
        float coordsz = dot(coords, coords);
        BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(tex, coords + st);
        BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
        float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .5);
        accumColor += colorFetch * tmpWeight;
        accumWeight += tmpWeight;
    }
    return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast20(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
    BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(tex, st);
    BILATERALBLURFAST_TYPE accumColor = BILATERALBLURFAST_TYPE(0.);
    float accumWeight = 0.;

    float scaleY2 = (BILATERALBLURFAST_NOISE_TEX_SIZE * pixel.y);
    vec2 scale = 10. * pixel;

    float yFetch = st.y * scaleY2;

    float sigma_r = .01 * (1. - smoothingFactor) + .1 * smoothingFactor;
    float sigma_r2 = 1. / (2. * sigma_r * sigma_r);

    for (float i = 0.0; i < 20. && accumWeight <= 5.; i++) {
        vec2 coords = BILATERALBLURFAST_NOISE_FNC(vec2(i * 0.05, yFetch));
        coords = (coords - .5) * scale;
        float coordsz = dot(coords, coords);
        BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(tex, coords + st);
        BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
        float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .125);
        accumColor += colorFetch * tmpWeight;
        accumWeight += tmpWeight;
    }
    return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast30(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
    BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(tex, st);
    BILATERALBLURFAST_TYPE accumColor = BILATERALBLURFAST_TYPE(0.);
    float accumWeight = 0.;

    float scaleY2 = (BILATERALBLURFAST_NOISE_TEX_SIZE * pixel.y);
    vec2 scale = 15. * pixel;

    float yFetch = st.y * scaleY2;

    float sigma_r = .01 * (1. - smoothingFactor) + .1 * smoothingFactor;
    float sigma_r2 = 1. / (2. * sigma_r * sigma_r);

    for (float i = 0.0; i < 30. && accumWeight <= 7.5; i++) {
        vec2 coords = BILATERALBLURFAST_NOISE_FNC(vec2(i * .03333333333, yFetch));
        coords = (coords - .5) * scale;
        float coordsz = dot(coords, coords);
        BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(tex, coords + st);
        BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
        float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .05555555556);
        accumColor += colorFetch * tmpWeight;
        accumWeight += tmpWeight;
    }
    return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast40(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
    BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(tex, st);
    BILATERALBLURFAST_TYPE accumColor = BILATERALBLURFAST_TYPE(0.);
    float accumWeight = 0.;

    float scaleY2 = (BILATERALBLURFAST_NOISE_TEX_SIZE * pixel.y);
    vec2 scale = 20. * pixel;

    float yFetch = st.y * scaleY2;

    float sigma_r = .01 * (1. - smoothingFactor) + .1 * smoothingFactor;
    float sigma_r2 = 1. / (2. * sigma_r * sigma_r);

    for (float i = 0.0; i < 40. && accumWeight <= 10.; i++) {
        vec2 coords = BILATERALBLURFAST_NOISE_FNC(vec2(i * .025, yFetch));
        coords = (coords - .5) * scale;
        float coordsz = dot(coords, coords);
        BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(tex, coords + st);
        BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
        float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .03125);
        accumColor += colorFetch * tmpWeight;
        accumWeight += tmpWeight;
    }
    return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast50(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
    BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(tex, st);
    BILATERALBLURFAST_TYPE accumColor = BILATERALBLURFAST_TYPE(0.);
    float accumWeight = 0.;

    float scaleY2 = (BILATERALBLURFAST_NOISE_TEX_SIZE * pixel.y);
    vec2 scale = 25. * pixel;

    float yFetch = st.y * scaleY2;

    float sigma_r = .01 * (1. - smoothingFactor) + .1 * smoothingFactor;
    float sigma_r2 = 1. / (2. * sigma_r * sigma_r);

    for (float i = 0.0; i < 50. && accumWeight <= 12.5; i++) {
        vec2 coords = BILATERALBLURFAST_NOISE_FNC(vec2(i * .02, yFetch));
        coords = (coords - .5) * scale;
        float coordsz = dot(coords, coords);
        BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(tex, coords + st);
        BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
        float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .02);
        accumColor += colorFetch * tmpWeight;
        accumWeight += tmpWeight;
    }
    return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast60(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
    BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(tex, st);
    BILATERALBLURFAST_TYPE accumColor = BILATERALBLURFAST_TYPE(0.);
    float accumWeight = 0.;

    float scaleY2 = (BILATERALBLURFAST_NOISE_TEX_SIZE * pixel.y);
    vec2 scale = 30. * pixel;

    float yFetch = st.y * scaleY2;

    float sigma_r = .01 * (1. - smoothingFactor) + .1 * smoothingFactor;
    float sigma_r2 = 1. / (2. * sigma_r * sigma_r);

    for (float i = 0.0; i < 60. && accumWeight <= 15.; i++) {
        vec2 coords = BILATERALBLURFAST_NOISE_FNC(vec2(i * .01666666667, yFetch));
        coords = (coords - .5) * scale;
        float coordsz = dot(coords, coords);
        BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(tex, coords + st);
        BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
        float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .01388888889);
        accumColor += colorFetch * tmpWeight;
        accumWeight += tmpWeight;
    }
    return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

#endif
