/*
contributors: Patricio Gonzalez Vivo
description: Pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
use: <vec3|vec4> rgb2YPbPr(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_RGB2PBPR
#define MAT_RGB2PBPR
#ifdef YPBPR_SDTV
const mat3 RGB2PBPR = mat3( 
    0.299, -0.169,  0.5,
    0.587, -0.331, -0.419,
    0.114,  0.5,   -0.081
);
#else
const mat3 RGB2PBPR = mat3( 
    0.2126, -0.1145721060573399,   0.5,
    0.7152, -0.3854278939426601,  -0.4541529083058166,
    0.0722,  0.5,                 -0.0458470916941834
);
#endif
#endif

#ifndef FNC_RGB2YPBPR
#define FNC_RGB2YPBPR
vec3 rgb2YPbPr(const in vec3 rgb) { return RGB2PBPR * rgb; }
vec4 rgb2YPbPr(const in vec4 rgb) { return vec4(rgb2YPbPr(rgb.rgb),rgb.a); }
#endif
