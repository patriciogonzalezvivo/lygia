/*
contributors: Patricio Gonzalez Vivo
description: Modify brightness and contrast
use: brightnessContrast(<float|vec3|vec4> color, <float> brightness, <float> amcontrastount)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_brightnessContrast.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_BRIGHTNESSCONTRAST
#define FNC_BRIGHTNESSCONTRAST
float brightnessContrast( float v, float b, float c ) { return ( v - 0.5 ) * c + 0.5 + b; }
vec3 brightnessContrast( vec3 v, float b, float c ) { return ( v - 0.5 ) * c + 0.5 + b; }
vec4 brightnessContrast( vec4 v, float b, float c ) { return vec4(( v.rgb - 0.5 ) * c + 0.5 + b, v.a); }
#endif