import lygia::space::sqTile::sqTile;

/*
contributors: Patricio Gonzalez Vivo
description: 'Return a black or white in a square checker patter'
use:
    - <vec4> checkerTile(<vec4> tile)
    - <vec4> checkerTile(<vec2> st [, <vec2> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn checkerTile(t: vec4f) -> f32 {
    let c = t.zw % vec2f(2.0);
    return abs(c.x - c.y);
}

fn checkerTile2(v: vec2f) -> f32 {
    return checkerTile(sqTile(v));
}

fn checkerTile_s(v: vec2f, s: f32) -> f32 {
    return checkerTile2(v * s);
}

fn checkerTile_s2(v: vec2f, s: vec2f) -> f32 {
    return checkerTile2(v * s);
}
