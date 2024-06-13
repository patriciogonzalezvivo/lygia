/*
contributors: Patricio Gonzalez Vivo
description: sampling function to make a texture behave like GL_NEAREST
use: nearest(float2 st, <float2> resolution)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_NEAREST
#define FNC_NEAREST
float2 nearest(in float2 st, in float2 resolution) {
    float2 offset = .5 / (resolution - 1.);
    return floor(st * resolution) / resolution + offset;
}
#endif