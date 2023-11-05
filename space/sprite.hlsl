/*
contributors: Patricio Gonzalez Vivo
description: returns a coordinate of a sprite cell
use: <float2> sprite(<float2> st, <float2> grid, <float> index)
*/

#ifndef FNC_SPRITE
#define FNC_SPRITE

float2 sprite(float2 st, float2 grid, float index) {
    float2 f = 1.0/grid;
    float2 cell = float2(floor(index), floor(index * f.x) );
    cell.y = grid.y - cell.y;
    return frac( (st+cell)*f );
}

#endif