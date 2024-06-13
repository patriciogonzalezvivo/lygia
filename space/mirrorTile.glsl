#include "sqTile.glsl"
#include "../math/mirror.glsl"

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

#ifndef FNC_MIRRORTILE
#define FNC_MIRRORTILE

vec4 mirrorTile(vec4 t) { return vec4( mirror(t.xy + t.zw), t.z, t.w); }
vec4 mirrorTile(vec2 v) { return mirrorTile(sqTile(v)); }
vec4 mirrorTile(vec2 v, float s) { return mirrorTile(v * s); }
vec4 mirrorTile(vec2 v, vec2 s) { return mirrorTile(v * s); }

vec4 mirrorXTile(vec4 t) { return vec4( mirror(t.x + t.z), t.y, t.z, t.w); }
vec4 mirrorXTile(vec2 v) { return mirrorXTile(sqTile(v)); }
vec4 mirrorXTile(vec2 v, float s) { return mirrorXTile(v * s); }
vec4 mirrorXTile(vec2 v, vec2 s) { return mirrorXTile(v * s); }

vec4 mirrorYTile(vec4 t) { return vec4( t.x, mirror(t.y + t.w), t.z, t.w); }
vec4 mirrorYTile(vec2 v) { return mirrorYTile(sqTile(v)); }
vec4 mirrorYTile(vec2 v, float s) { return mirrorYTile(v * s); }
vec4 mirrorYTile(vec2 v, vec2 s) { return mirrorYTile(v * s); }

#endif