#include "tonemap/aces.wgsl"
#include "tonemap/debug.wgsl"
#include "tonemap/filmic.wgsl"
#include "tonemap/linear.wgsl"
#include "tonemap/reinhard.wgsl"
#include "tonemap/reinhardJodie.wgsl"
#include "tonemap/uncharted.wgsl"
#include "tonemap/uncharted2.wgsl"
#include "tonemap/unreal.wgsl"

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

//     #define TONEMAP_FNC     tonemapUnreal
    // #define TONEMAP_FNC     tonemapDebug
    // #define TONEMAP_FNC     tonemapFilmic
    // #define TONEMAP_FNC     tonemapACES
    // #define TONEMAP_FNC     tonemapUncharted2
    // #define TONEMAP_FNC     tonemapUncharted
//     #define TONEMAP_FNC     tonemapReinhardJodie
    // #define TONEMAP_FNC     tonemapReinhard
    // #define TONEMAP_FNC     tonemapUnreal
    // #define TONEMAP_FNC     tonemapLinear

fn tonemap3(v: vec3f) -> vec3f { return TONEMAP_FNC(v); }
fn tonemap4(v: vec4f) -> vec4f { return TONEMAP_FNC(v); }
