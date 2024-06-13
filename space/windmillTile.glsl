#include "../math/const.glsl"
#include "rotate.glsl"
#include "sqTile.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Rotate tiles (in a squared grid pattern) by 45 degrees'
use:
    - <vec4> windmillTile(<vec4> tiles[, <float> fullturn = TAU])
    - <vec2> windmillTile(<vec2> st [, <float|vec2> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WINDMILLTILE
#define FNC_WINDMILLTILE
vec4 windmillTile(vec4 t, float turn) {
    float a = ( abs(mod(t.z, 2.0)-
                    mod(t.w, 2.0))+
                mod(t.w, 2.0) * 2.0)*
                0.25;
    return vec4(rotate(t.xy, a * turn), t.zw);
}

vec4 windmillTile(vec4 t) {
    return windmillTile(t, TAU);
}

vec4 windmillTile(vec2 v) {
    return windmillTile(sqTile(v));
}

vec4 windmillTile(vec2 v, float s) {
    return windmillTile(v * s);
}

vec4 windmillTile(vec2 v, vec2 s) {
    return windmillTile(v * s);
}
#endif