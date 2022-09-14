/*
original_author: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
use: YPbPr2RGB(<vec3|vec4> color)
*/

#ifndef FNC_YPBPR2RGB
#define FNC_YPBPR2RGB

#ifdef YPBPR_SDTV
const mat3 YPbPr2rgb_mat = mat3( 
    1.,     1.,       1.,
    0.,     -.344,    1.772,
    1.402,  -.714,    0.
);
#else
const mat3 YPbPr2rgb_mat = mat3( 
    1.,     1.,       1.,
    0.,     -.187,    1.856,
    1.575,  -.468,    0.
);
#endif

vec3 YPbPr2rgb(in vec3 rgb) {
    return YPbPr2rgb_mat * rgb;
}

vec4 YPbPr2rgb(in vec4 rgb) {
    return vec4(YPbPr2rgb(rgb.rgb),rgb.a);
}
#endif