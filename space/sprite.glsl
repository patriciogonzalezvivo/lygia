/*
contributors: Patricio Gonzalez Vivo
description: returns a coordinate of a sprite cell
use: <vec2> sprite(<vec2> st, <vec2> grid, <float> index)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SPRITE
#define FNC_SPRITE

vec2 sprite(vec2 st, vec2 grid, float index) {
    index += grid.x; 
    vec2 f = 1.0/grid;
    vec2 cell = vec2(floor(index), grid.y - floor(index * f.x) );
    // cell.y = grid.y - cell.y;
    return fract( (st+cell)*f );
}

#endif