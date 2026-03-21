#include "../math/const.wgsl"
#include "rotate.wgsl"
#include "sqTile.wgsl"

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

fn windmillTile4(t: vec4f, turn: f32) -> vec4f {
    float a = ( abs(mod(t.z, 2.0)-
                    mod(t.w, 2.0))+
                mod(t.w, 2.0) * 2.0)*
                0.25;
    return vec4f(rotate(t.xy, a * turn), t.zw);
}

fn windmillTile4a(t: vec4f) -> vec4f {
    return windmillTile(t, TAU);
}

fn windmillTile2(v: vec2f) -> vec4f {
    return windmillTile(sqTile(v));
}

fn windmillTile2a(v: vec2f, s: f32) -> vec4f {
    return windmillTile(v * s);
}

fn windmillTile2b(v: vec2f, s: vec2f) -> vec4f {
    return windmillTile(v * s);
}
