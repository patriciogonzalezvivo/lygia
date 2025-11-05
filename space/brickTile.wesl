import lygia::space::sqTile::sqTile;

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

fn brickTile(t: vec4f) -> vec4f {
    var result = t;
    result.x += (t.w % 2.0) * 0.5;
    result.z = floor(result.z + result.x);
    result.x = fract(result.x);
    return result;
}

fn brickTile2(st: vec2f) -> vec4f {
    return brickTile(sqTile(st));
}

fn brickTile_s(st: vec2f, s: f32) -> vec4f {
    return brickTile2(st * s);
}

fn brickTile_s2(st: vec2f, s: vec2f) -> vec4f {
    return brickTile2(st * s);
}
