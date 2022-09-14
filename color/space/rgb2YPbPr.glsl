/*
original_author: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
use: rgb2YPbPr(<vec3|vec4> color)
*/

#ifndef FNC_RGB2YPBPR
#define FNC_RGB2YPBPR

#ifdef YPBPR_SDTV
const mat3 rgb2YPbPr_mat = mat3( 
    .299, -.169,  .5,
    .587, -.331, -.419,
    .114,  .5,   -.081
);
#else
const mat3 rgb2YPbPr_mat = mat3( 
    0.2126, -0.1145721060573399,   0.5,
    0.7152, -0.3854278939426601,  -0.4541529083058166,
    0.0722,  0.5,                 -0.0458470916941834
);
#endif

vec3 rgb2YPbPr(in vec3 rgb) {
    return rgb2YPbPr_mat * rgb;
}

vec4 rgb2YPbPr(in vec4 rgb) {
    return vec4(rgb2YPbPr(rgb.rgb),rgb.a);
}
#endif
