/*
contributors: Patricio Gonzalez Vivo
description: |
   Convert a color in YIQ to linear RGB color. 
   From https://en.wikipedia.org/wiki/YIQ
use: <vec3|vec4> yiq2rgb(<vec3|vec4> color)
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
