/*
contributors: Patricio Gonzalez Vivo
description: returns a coordinate of a sprite cell
use: <vec2f> sprite(<vec2f> st, <vec2f> grid, <f32> index)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sprite(st: vec2f, grid: vec2f, index: f32) -> vec2f {
    let i = index + grid.x; 
    let f = 1.0 / grid;
    let cell = vec2f(floor(i), grid.y - floor(i * f.x));
    // cell.y = grid.y - cell.y;
    return fract((st + cell) * f);
}
