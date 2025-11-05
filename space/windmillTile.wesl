import lygia::math::consts::TAU;
import lygia::space::rotate::rotate;
import lygia::space::sqTile::sqTile;

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

fn windmillTile_turn(t: vec4f, turn: f32) -> vec4f {
    let a = (abs((t.z % 2.0) - (t.w % 2.0)) + (t.w % 2.0) * 2.0) * 0.25;
    return vec4f(rotate(t.xy, a * turn), t.zw);
}

fn windmillTile(t: vec4f) -> vec4f {
    return windmillTile_turn(t, TAU);
}

fn windmillTile2(v: vec2f) -> vec4f {
    return windmillTile(sqTile(v));
}

fn windmillTile_s(v: vec2f, s: f32) -> vec4f {
    return windmillTile2(v * s);
}

fn windmillTile_s2(v: vec2f, s: vec2f) -> vec4f {
    return windmillTile2(v * s);
}
