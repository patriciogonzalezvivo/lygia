/*
contributors: Patricio Gonzalez Vivo
description: "Converts a color in YIQ to linear RGB color. \nFrom https://en.wikipedia.org/wiki/YIQ\n"
use: <vec3|vec4> yiq2rgb(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_YIQ2RGB
#define MAT_YIQ2RGB
const mat3 YIQ2RGB = mat3(  1.0,  0.9469,  0.6235, 
                            1.0, -0.2747, -0.6357, 
                            1.0, -1.1085,  1.7020 );
#endif

#ifndef FNC_YIQ2RGB
#define FNC_YIQ2RGB
vec3 yiq2rgb(const in vec3 yiq) { return YIQ2RGB * yiq; }
vec4 yiq2rgb(const in vec4 yiq) { return vec4(yiq2rgb(yiq.rgb), yiq.a); }
#endif
