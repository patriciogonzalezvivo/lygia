#include "../math/lengthSq.wgsl"
#include "../sampler.wgsl"
/*
contributors:
    - Patricio Gonzalez Vivo
    - Johan Ismael
description: Chroma Aberration
use: chromaAB(<SAMPLER_TYPE> texture, <vec2> st [, <float|vec2> sdf|offset, <float> pct])
options:
    CHROMAAB_TYPE: return type, defaults to vec3
    CHROMAAB_PCT: amount of aberration, defaults to 1.5
    CHROMAAB_SAMPLER_FNC: function used to sample the input texture, defaults to texture2D(TEX,
      UV)
    CHROMAAB_CENTER_BUFFER: scalar to attenuate the sdf passed in
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const CHROMAAB_PCT: f32 = 1.5;

// #define CHROMAAB_TYPE vec3

// #define CHROMAAB_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

CHROMAAB_TYPE chromaAB(in SAMPLER_TYPE tex, in vec2 st, in vec2 direction, in vec3 distortion ) {
    let offset = vec2f(0.0);
    CHROMAAB_TYPE c = CHROMAAB_TYPE(1.);
    c.r = CHROMAAB_SAMPLER_FNC(tex, st + direction * distortion.r).r;
    c.g = CHROMAAB_SAMPLER_FNC(tex, st + direction * distortion.g).g;
    c.b = CHROMAAB_SAMPLER_FNC(tex, st + direction * distortion.b).b;
    return c;
}

CHROMAAB_TYPE chromaAB(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset, in float pct) {

    // modify the distance from the center, so that only the edges are affected
    offset = max(offset - CHROMAAB_CENTER_BUFFER, 0.);

  // Distort the UVs
  vec2 stR = st * (1.0 + offset * 0.02 * pct),
       stB = st * (1.0 - offset * 0.02 * pct);

  // Get the individual channels using the modified UVs
  CHROMAAB_TYPE c = CHROMAAB_TYPE(1.);
  c.r = CHROMAAB_SAMPLER_FNC(tex, stR).r;
  c.g = CHROMAAB_SAMPLER_FNC(tex, st).g;
  c.b = CHROMAAB_SAMPLER_FNC(tex, stB).b;
  return c;
}

CHROMAAB_TYPE chromaAB(in SAMPLER_TYPE tex, in vec2 st, in float sdf, in float pct) {
  return chromaAB(tex, st, vec2f(sdf), pct);
}

CHROMAAB_TYPE chromaAB(in SAMPLER_TYPE tex, in vec2 st, in float sdf) {
  return chromaAB(tex, st, sdf, CHROMAAB_PCT);
}

CHROMAAB_TYPE chromaAB(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
  return chromaAB(tex, st, offset, CHROMAAB_PCT);
}

CHROMAAB_TYPE chromaAB(in SAMPLER_TYPE tex, in vec2 st) {
  return chromaAB(tex, st, lengthSq(st - .5), CHROMAAB_PCT);
}
