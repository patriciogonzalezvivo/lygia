/*
function: nearest
contributors: Patricio Gonzalez Vivo
description: sampling function to make a texture behave like GL_NEAREST 
use: nearest(float2 st, <float2> resolution)
*/

#ifndef FNC_NEAREST
#define FNC_NEAREST
float2 nearest(in float2 st, in float2 resolution) {
    float2 offset = .5 / (resolution - 1.);
    return floor(st * resolution) / resolution + offset;
}
#endif