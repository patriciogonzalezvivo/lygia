#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    convertes QUILT of tiles into something the LookingGlass Volumetric display can render
use: sampleQuilt(<SAMPLER_TYPE> texture, <vec4> calibration, <vec3> tile, <vec2> st, <vec2> resolution)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEQUILT_FLIPSUBP: null
    - SAMPLEQUILT_SAMPLER_FNC(POS_UV): Function used to sample into the normal map texture, defaults to texture2D(tex,POS_UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SAMPLEQUILT_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

fn mapQuilt(tile: vec3f, pos: vec2f, a: f32) -> vec2f {
    let tile2 = tile.xy - 1.0;
    let dir = vec2f(-1.0);

    a = fract(a) * tile.y;
    tile2.y += dir.y * floor(a);
    a = fract(a) * tile.x;
    tile2.x += dir.x * floor(a);
    return (tile2 + pos) / tile.xy;
}

fn sampleQuilt(tex: SAMPLER_TYPE, calibration: vec4f, tile: vec3f, st: vec2f, resolution: vec2f) -> vec3f {
    let pitch = -resolution.x / calibration.x  * calibration.y * sin(atan(abs(calibration.z)));
    let tilt = resolution.y / (resolution.x * calibration.z);
    let subp = 1.0 / (3.0 * resolution.x);
    let subp2 = subp * pitch;

    let a = (-st.x - st.y * tilt) * pitch - calibration.w;

    let color = vec3f(0.0);
    color.r = SAMPLEQUILT_SAMPLER_FNC(tex, mapQuilt(tile, st, a-2.0*subp2) ).r;
    color.g = SAMPLEQUILT_SAMPLER_FNC(tex, mapQuilt(tile, st, a-subp2) ).g;
    color.b = SAMPLEQUILT_SAMPLER_FNC(tex, mapQuilt(tile, st, a) ).b;
    color.r = SAMPLEQUILT_SAMPLER_FNC(tex, mapQuilt(tile, st, a) ).r;
    color.g = SAMPLEQUILT_SAMPLER_FNC(tex, mapQuilt(tile, st, a-subp2) ).g;
    color.b = SAMPLEQUILT_SAMPLER_FNC(tex, mapQuilt(tile, st, a-2.0*subp2) ).b;
    return color;
}
