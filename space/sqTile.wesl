/*
contributors: Patricio Gonzalez Vivo
description: make some square tiles. XY provide coords inside of the tile. ZW provides tile coords
use: <vec4> hexTile(<vec2> st [, <float> scale])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sqTile(st: vec2f) -> vec4f {
    return vec4f(fract(st), floor(st));
}

fn sqTile_scale(st: vec2f, scale: f32) -> vec4f {
    return sqTile(st * scale);
}
