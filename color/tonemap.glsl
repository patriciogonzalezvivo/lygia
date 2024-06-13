#include "tonemap/aces.glsl"
#include "tonemap/debug.glsl"
#include "tonemap/filmic.glsl"
#include "tonemap/linear.glsl"
#include "tonemap/reinhard.glsl"
#include "tonemap/reinhardJodie.glsl"
#include "tonemap/uncharted.glsl"
#include "tonemap/uncharted2.glsl"
#include "tonemap/unreal.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Tone maps the specified RGB color (meaning convert from HDR to LDR) inside the range [0..~8] to [0..1]. The input must be in linear HDR pre-exposed.
use: tonemap(<vec3|vec4> rgb)
options:
    - TONEMAP_FNC: |
        tonemapLinear, tonemapReinhard, tonemapUnreal, tonemapACES, tonemapDebug,
        tonemapUncharter
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef TONEMAP_FNC
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI)
    #define TONEMAP_FNC     tonemapUnreal
#else
    // #define TONEMAP_FNC     tonemapDebug
    // #define TONEMAP_FNC     tonemapFilmic
    // #define TONEMAP_FNC     tonemapACES
    // #define TONEMAP_FNC     tonemapUncharted2
    // #define TONEMAP_FNC     tonemapUncharted
    #define TONEMAP_FNC     tonemapReinhardJodie
    // #define TONEMAP_FNC     tonemapReinhard
    // #define TONEMAP_FNC     tonemapUnreal
    // #define TONEMAP_FNC     tonemapLinear
#endif
#endif

#ifndef FNC_TONEMAP
#define FNC_TONEMAP

vec3 tonemap(const vec3 v) { return TONEMAP_FNC(v); }
vec4 tonemap(const vec4 v) { return TONEMAP_FNC(v); }

#endif