#include "../../math/saturate.glsl"
#include "../../math/decimate.glsl"
#include "../luma.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Dither using a 8x8 Bayer matrix
use: 
 - <vec4|vec3|float> ditherBayer(<vec4|vec3|float> value, <vec2> st, <float> time)
 - <vec4|vec3|float> ditherBayer(<vec4|vec3|float> value, <vec2> st)
 - <float> ditherBayer(<vec2> xy)
options:
    - DITHERBAKER_LUT(COLOR): function that returns a vec3 with the color to use for the dithering
examples:
    - /shaders/color_dither_bayer.frag
*/

#ifndef FNC_DITHERBAYER
#define FNC_DITHERBAYER

#ifndef DITHERBAKER_LUT
#define DITHERBAKER_LUT(COLOR) vec3(decimate(pow(luma(COLOR) * 1.3, 1.2), 4.0))
#endif

#if defined(PLATFORM_WEBGL)
float ditherBayer(const in vec2 xy) {
    float x = mod(xy.x, 8.0);
    float y = mod(xy.y, 8.0);
    return mix(mix(mix(mix(mix(mix(0.0,32.0,step(1.0,y)),mix(8.0,40.0,step(3.0,y)),step(2.0,y)),mix(mix(2.0,34.0,step(5.0,y)),mix(10.0,42.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),mix(mix(mix(48.0,16.0,step(1.0,y)),mix(56.0,24.0,step(3.0,y)),step(2.0,y)),mix(mix(50.0,18.0,step(5.0,y)),mix(58.0,26.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),step(1.0,x)),mix(mix(mix(mix(12.0,44.0,step(1.0,y)),mix(4.0,36.0,step(3.0,y)),step(2.0,y)),mix(mix(14.0,46.0,step(5.0,y)),mix(6.0,38.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),mix(mix(mix(60.0,28.0,step(1.0,y)),mix(52.0,20.0,step(3.0,y)),step(2.0,y)),mix(mix(62.0,30.0,step(5.0,y)),mix(54.0,22.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),step(3.0,x)),step(2.0,x)),mix(mix(mix(mix(mix(3.0,35.0,step(1.0,y)),mix(11.0,43.0,step(3.0,y)),step(2.0,y)),mix(mix(1.0,33.0,step(5.0,y)),mix(9.0,41.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),mix(mix(mix(51.0,19.0,step(1.0,y)),mix(59.0,27.0,step(3.0,y)),step(2.0,y)),mix(mix(49.0,17.0,step(5.0,y)),mix(57.0,25.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),step(5.0,x)),mix(mix(mix(mix(15.0,47.0,step(1.0,y)),mix(7.0,39.0,step(3.0,y)),step(2.0,y)),mix(mix(13.0,45.0,step(5.0,y)),mix(5.0,37.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),mix(mix(mix(63.0,31.0,step(1.0,y)),mix(55.0,23.0,step(3.0,y)),step(2.0,y)),mix(mix(61.0,29.0,step(5.0,y)),mix(53.0,21.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),step(7.0,x)),step(6.0,x)),step(4.0,x)) / 255.0;
}

#else

float ditherBayer(const in vec2 xy) {
    float kern[64];
    kern[  0] = 0.000; kern[  1] = 0.125; kern[  2] = 0.031; kern[  3] = 0.156; kern[  4] = 0.007; kern[  5] = 0.133; kern[  6] = 0.039; kern[  7] = 0.164; 
    kern[  8] = 0.188; kern[  9] = 0.062; kern[ 10] = 0.219; kern[ 11] = 0.094; kern[ 12] = 0.196; kern[ 13] = 0.070; kern[ 14] = 0.227; kern[ 15] = 0.101;
    kern[ 16] = 0.047; kern[ 17] = 0.172; kern[ 18] = 0.015; kern[ 19] = 0.141; kern[ 20] = 0.054; kern[ 21] = 0.180; kern[ 22] = 0.023; kern[ 23] = 0.149; 
    kern[ 24] = 0.235; kern[ 25] = 0.109; kern[ 26] = 0.203; kern[ 27] = 0.078; kern[ 28] = 0.243; kern[ 29] = 0.117; kern[ 30] = 0.211; kern[ 31] = 0.086;
    kern[ 32] = 0.011; kern[ 33] = 0.137; kern[ 34] = 0.043; kern[ 35] = 0.168; kern[ 36] = 0.003; kern[ 37] = 0.129; kern[ 38] = 0.035; kern[ 39] = 0.160;
    kern[ 40] = 0.200; kern[ 41] = 0.074; kern[ 42] = 0.231; kern[ 43] = 0.105; kern[ 44] = 0.192; kern[ 45] = 0.066; kern[ 46] = 0.223; kern[ 47] = 0.098;
    kern[ 48] = 0.058; kern[ 49] = 0.184; kern[ 50] = 0.027; kern[ 51] = 0.152; kern[ 52] = 0.050; kern[ 53] = 0.176; kern[ 54] = 0.019; kern[ 55] = 0.145;
    kern[ 56] = 0.247; kern[ 57] = 0.121; kern[ 58] = 0.215; kern[ 59] = 0.090; kern[ 60] = 0.239; kern[ 61] = 0.113; kern[ 62] = 0.207; kern[ 63] = 0.082;
    return kern[int(mod(xy.x, 8.0)) + (int(mod(xy.y, 8.0)) * 8)];
}
#endif

vec3 ditherBayer(const in vec3 color, const in vec2 xy) {
    return DITHERBAKER_LUT( saturate(color + ditherBayer(xy)) );
}

vec4 ditherBayer(const in vec4 color, const in vec2 xy) { return vec4(ditherBayer(color.rgb, xy), color.a); }
vec3 ditherBayer(const in vec3 color, const in vec2 xy, float time) { return ditherBayer(color, xy); }
vec4 ditherBayer(const in vec4 color, const in vec2 xy, float time) { return ditherBayer(color, xy); }
#endif