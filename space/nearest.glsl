/*
function: nearest
original_author: Patricio Gonzalez Vivo
description: sampling function to make a texture behave like GL_NEAREST 
use: nearest(vec2 st, <vec2> resolution)
*/

#ifndef FNC_NEAREST
#define FNC_NEAREST
vec2 nearest(in vec2 st, in vec2 resolution) {
    vec2 offset = .5 / (resolution - 1.);
    return floor(st * resolution) / resolution + offset;
}
#endif