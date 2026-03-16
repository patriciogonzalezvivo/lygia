#include "sqTile.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Brick a squared pattern'
use:
    - <vec2> brickTile(<vec2> st [, <float> scale])
    - <vec4> brickTile(<vec4> tiles)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn brickTile4(t: vec4f) -> vec4f {
    t.x += mod(t.w,2.)*.5;
    t.z = floor(t.z+t.x);
    t.x = fract(t.x);
    return t;
}

fn brickTile2(st: vec2f) -> vec4f {
    return brickTile(sqTile(st));
}

fn brickTile2a(st: vec2f, s: f32) -> vec4f {
    return brickTile(st * s);
}

fn brickTile2b(st: vec2f, s: vec2f) -> vec4f {
    return brickTile(st * s);
}
