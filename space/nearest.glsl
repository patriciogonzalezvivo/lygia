/*
function: nearest
contributors: Patricio Gonzalez Vivo
description: sampling function to make a texture behave like GL_NEAREST 
use: nearest(vec2 st, <vec2> res)
*/

#ifndef FNC_NEAREST
#define FNC_NEAREST
vec2 nearest(in vec2 v, in vec2 res) {
    vec2 offset = 0.5 / (res - 1.0);
    return floor(v * res) / res + offset;
}
#endif