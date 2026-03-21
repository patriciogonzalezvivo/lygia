#include "sqTile.wgsl"
#include "../math/mirror.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: mirror a tiles pattern
use:
    - <vec4> mirrorTile(<vec4> tile)
    - <vec4> mirrorTile(<vec2> st [, <float|vec2> scale])
    - <vec4> mirrorXTile(<vec4> tile)
    - <vec4> mirrorXTile(<vec2> st [, <float|vec2> scale])
    - <vec4> mirrorYTile(<vec4> tile)
    - <vec4> mirrorYTile(<vec2> st [, <float|vec2> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn mirrorTile4(t: vec4f) -> vec4f { return vec4f( mirror(t.xy + t.zw), t.z, t.w); }
fn mirrorTile2(v: vec2f) -> vec4f { return mirrorTile(sqTile(v)); }
fn mirrorTile2a(v: vec2f, s: f32) -> vec4f { return mirrorTile(v * s); }
fn mirrorTile2b(v: vec2f, s: vec2f) -> vec4f { return mirrorTile(v * s); }

fn mirrorXTile4(t: vec4f) -> vec4f { return vec4f( mirror(t.x + t.z), t.y, t.z, t.w); }
fn mirrorXTile2(v: vec2f) -> vec4f { return mirrorXTile(sqTile(v)); }
fn mirrorXTile2a(v: vec2f, s: f32) -> vec4f { return mirrorXTile(v * s); }
fn mirrorXTile2b(v: vec2f, s: vec2f) -> vec4f { return mirrorXTile(v * s); }

fn mirrorYTile4(t: vec4f) -> vec4f { return vec4f( t.x, mirror(t.y + t.w), t.z, t.w); }
fn mirrorYTile2(v: vec2f) -> vec4f { return mirrorYTile(sqTile(v)); }
fn mirrorYTile2a(v: vec2f, s: f32) -> vec4f { return mirrorYTile(v * s); }
fn mirrorYTile2b(v: vec2f, s: vec2f) -> vec4f { return mirrorYTile(v * s); }
