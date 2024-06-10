/*
contributors: Patricio Gonzalez Vivo
description: 'Get''s the luminosity from sRGB. Based on from Rec601 luma (see https://en.wikipedia.org/wiki/Grayscale)'
use: <float> rgb2luma(<vec3|vec4> srgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SRGB2LUMA
#define FNC_SRGB2LUMA
float srgb2luma(const in vec3 srgb) { return dot(srgb, vec3(0.299, 0.587, 0.114)); }
float srgb2luma(const in vec4 srgb) { return rgb2luma(srgb.rgb); }
#endif
