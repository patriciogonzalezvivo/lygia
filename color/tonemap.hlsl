#include "tonemap/aces.hlsl"
#include "tonemap/debug.hlsl"
#include "tonemap/linear.hlsl"
#include "tonemap/reinhard.hlsl"
#include "tonemap/uncharted.hlsl"
#include "tonemap/unreal.hlsl"

#include "tonemap/aces.hlsl"
#include "tonemap/debug.hlsl"
#include "tonemap/filmic.hlsl"
#include "tonemap/linear.hlsl"
#include "tonemap/reinhard.hlsl"
#include "tonemap/reinhardJodie.hlsl"
#include "tonemap/uncharted.hlsl"
#include "tonemap/uncharted2.hlsl"
#include "tonemap/unreal.hlsl"
/*
contributors: Patricio Gonzalez Vivo
description: Tone maps the specified RGB color (meaning convert from HDR to LDR) inside the range [0..~8] to [0..1]. The input must be in linear HDR pre-exposed.
use: tonemap(<float3|float4> rgb)
options:
    - TONEMAP_FNC: | 
        tonemapLinear, tonemapReinhard, tonemapUnreal, tonemapACES, tonemapDebug,
        tonemapUncharter
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef TONEMAP_FNC
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
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

float3 tonemap(const float3 color) { return TONEMAP_FNC(color); }
float4 tonemap(const float4 color) { return TONEMAP_FNC(color); }

#endif