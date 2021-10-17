#include "../../space/nearest.glsl"

/*
author: Patricio Gonzalez Vivo
description: one dimensional bilateral Blur that use a blue noise texture to sample a kernel with out introducing biases
use: bilateralBlurFast(<sampler2D> texture, <vec2> st, <vec2> pixelSize, <float> smoothingFactor,  <float> kernelSize)
options:
  BILATERALBLURFAST_TYPE: default is vec3
  BILATERALBLURFAST_SAMPLER_FNC(POS_UV): default texture2D(tex, POS_UV)
  BILATERALBLURFAST_NOISE_TEX_SIZE: blue noise texture size. Default 64.0
  BILATERALBLURFAST_SIGMA_R: sigma defualt .075
  BILATERALBLURFAST_NOISE_FNC: functions use to sample the blue noise texture
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef BILATERALBLURFAST_TYPE
#define BILATERALBLURFAST_TYPE vec3
#endif

#ifndef BILATERALBLURFAST_SAMPLER_FNC
#define BILATERALBLURFAST_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV).rgb
#endif

#ifndef BILATERALBLURFAST_NOISE_TEX_SIZE
#define BILATERALBLURFAST_NOISE_TEX_SIZE 64.0
#endif

#ifndef BILATERALBLURFAST_SIGMA_R
#define BILATERALBLURFAST_SIGMA_R .075
#endif

#ifndef BILATERALBLURFAST_NOISE_FNC
#define BILATERALBLURFAST_NOISE_FNC(POS_UV) texture2D(poissonNoise, nearest(fract(POS_UV),vec2(BILATERALBLURFAST_NOISE_TEX_SIZE))).xy
#endif

#ifndef FNC_BILATERALBLURFAST
#define FNC_BILATERALBLURFAST

#if define(PLATFORM_RPI)
BILATERALBLURFAST_TYPE bilateralBlurFast(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor, const float sigma_s) {
  BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(st);
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
    BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(coords + st);
    BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
    float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * sigma_s2);
    accumColor += colorFetch * tmpWeight;
    accumWeight += tmpWeight;
  }
  return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}
#endif

BILATERALBLURFAST_TYPE bilateralBlurFast10(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
  BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(st);
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
    BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(coords + st);
    BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
    float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .5);
    accumColor += colorFetch * tmpWeight;
    accumWeight += tmpWeight;
  }
  return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast20(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
  BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(st);
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
    BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(coords + st);
    BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
    float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .125);
    accumColor += colorFetch * tmpWeight;
    accumWeight += tmpWeight;
  }
  return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast30(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
  BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(st);
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
    BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(coords + st);
    BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
    float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .05555555556);
    accumColor += colorFetch * tmpWeight;
    accumWeight += tmpWeight;
  }
  return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast40(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
  BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(st);
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
    BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(coords + st);
    BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
    float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .03125);
    accumColor += colorFetch * tmpWeight;
    accumWeight += tmpWeight;
  }
  return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast50(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
  BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(st);
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
    BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(coords + st);
    BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
    float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .02);
    accumColor += colorFetch * tmpWeight;
    accumWeight += tmpWeight;
  }
  return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

BILATERALBLURFAST_TYPE bilateralBlurFast60(in sampler2D tex, in vec2 st, in vec2 pixel, in float smoothingFactor) {
  BILATERALBLURFAST_TYPE colorRef = BILATERALBLURFAST_SAMPLER_FNC(st);
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
    BILATERALBLURFAST_TYPE colorFetch = BILATERALBLURFAST_SAMPLER_FNC(coords + st);
    BILATERALBLURFAST_TYPE colorDist = colorFetch - colorRef;
    float tmpWeight = exp(-dot(colorDist, colorDist) * sigma_r2 - coordsz * .01388888889);
    accumColor += colorFetch * tmpWeight;
    accumWeight += tmpWeight;
  }
  return accumWeight > 0.0 ? accumColor / accumWeight : colorRef;
}

#endif
