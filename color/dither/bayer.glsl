#include "../../math/decimate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Dither using a 8x8 Bayer matrix
use:
    - <vec4|vec3|float> ditherBayer(<vec4|vec3|float> value, <vec2> st, <float> time)
    - <vec4|vec3|float> ditherBayer(<vec4|vec3|float> value, <vec2> st)
    - <float> ditherBayer(<vec2> xy)
options:
    - DITHER_BAKER_COORD
examples:
    - /shaders/color_dither_bayer.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DITHER_BAKER_COORD
#define DITHER_BAKER_COORD gl_FragCoord.xy
#endif

#ifndef DITHER_BAYER_PRECISION
#ifdef DITHER_PRECISION
#define DITHER_BAYER_PRECISION DITHER_PRECISION
#else
#define DITHER_BAYER_PRECISION 256
#endif
#endif

#ifndef FNC_DITHER_BAYER
#define FNC_DITHER_BAYER

#if defined(PLATFORM_WEBGL)
float ditherBayer(const in vec2 xy) {
    float x = mod(xy.x, 8.0);
    float y = mod(xy.y, 8.0);
    return  mix(mix(mix(mix(mix(mix(0.0,32.0,step(1.0,y)),mix(8.0,40.0,step(3.0,y)),step(2.0,y)),mix(mix(2.0,34.0,step(5.0,y)),mix(10.0,42.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),mix(mix(mix(48.0,16.0,step(1.0,y)),mix(56.0,24.0,step(3.0,y)),step(2.0,y)),mix(mix(50.0,18.0,step(5.0,y)),mix(58.0,26.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),step(1.0,x)),mix(mix(mix(mix(12.0,44.0,step(1.0,y)),mix(4.0,36.0,step(3.0,y)),step(2.0,y)),mix(mix(14.0,46.0,step(5.0,y)),mix(6.0,38.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),mix(mix(mix(60.0,28.0,step(1.0,y)),mix(52.0,20.0,step(3.0,y)),step(2.0,y)),mix(mix(62.0,30.0,step(5.0,y)),mix(54.0,22.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),step(3.0,x)),step(2.0,x)),mix(mix(mix(mix(mix(3.0,35.0,step(1.0,y)),mix(11.0,43.0,step(3.0,y)),step(2.0,y)),mix(mix(1.0,33.0,step(5.0,y)),mix(9.0,41.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),mix(mix(mix(51.0,19.0,step(1.0,y)),mix(59.0,27.0,step(3.0,y)),step(2.0,y)),mix(mix(49.0,17.0,step(5.0,y)),mix(57.0,25.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),step(5.0,x)),mix(mix(mix(mix(15.0,47.0,step(1.0,y)),mix(7.0,39.0,step(3.0,y)),step(2.0,y)),mix(mix(13.0,45.0,step(5.0,y)),mix(5.0,37.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),mix(mix(mix(63.0,31.0,step(1.0,y)),mix(55.0,23.0,step(3.0,y)),step(2.0,y)),mix(mix(61.0,29.0,step(5.0,y)),mix(53.0,21.0,step(7.0,y)),step(6.0,y)),step(4.0,y)),step(7.0,x)),step(6.0,x)),step(4.0,x)) / (64.0);
}

#else

float ditherBayer(const in vec2 xy) {
    float kern[64];
    kern[ 0] = 0.000; kern[ 1] = 0.500; kern[ 2] = 0.124; kern[ 3] = 0.624; kern[ 4] = 0.028; kern[ 5] = 0.532; kern[ 6] = 0.156; kern[ 7] = 0.656; 
    kern[ 8] = 0.752; kern[ 9] = 0.248; kern[10] = 0.876; kern[11] = 0.376; kern[12] = 0.784; kern[13] = 0.280; kern[14] = 0.908; kern[15] = 0.404; 
    kern[16] = 0.188; kern[17] = 0.688; kern[18] = 0.060; kern[19] = 0.564; kern[20] = 0.216; kern[21] = 0.720; kern[22] = 0.092; kern[23] = 0.596; 
    kern[24] = 0.940; kern[25] = 0.436; kern[26] = 0.812; kern[27] = 0.312; kern[28] = 0.972; kern[29] = 0.468; kern[30] = 0.844; kern[31] = 0.344; 
    kern[32] = 0.044; kern[33] = 0.548; kern[34] = 0.172; kern[35] = 0.672; kern[36] = 0.012; kern[37] = 0.516; kern[38] = 0.140; kern[39] = 0.640; 
    kern[40] = 0.800; kern[41] = 0.296; kern[42] = 0.924; kern[43] = 0.420; kern[44] = 0.768; kern[45] = 0.264; kern[46] = 0.892; kern[47] = 0.392; 
    kern[48] = 0.232; kern[49] = 0.736; kern[50] = 0.108; kern[51] = 0.608; kern[52] = 0.200; kern[53] = 0.704; kern[54] = 0.076; kern[55] = 0.580; 
    kern[56] = 0.988; kern[57] = 0.484; kern[58] = 0.860; kern[59] = 0.360; kern[60] = 0.956; kern[61] = 0.452; kern[62] = 0.828; kern[63] = 0.328;
    int index = int(mod(xy.x, 8.0)) + (int(mod(xy.y, 8.0)) * 8);
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++) if (i == index) return kern[i];
    #else
    return kern[index];
    #endif
}

#endif

vec3 ditherBayer(vec3 color, const in vec2 xy, const int pres) {
    float d = float(pres);
    vec3 decimated = decimate(color, d);
    vec3 diff = (color - decimated) * d;
    vec3 ditherPattern = vec3(ditherBayer(xy));
    return decimate(color + (step(ditherPattern, diff) / d), d);
}

float ditherBayer(const in float val, const in vec2 xy) { return ditherBayer(vec3(val), xy, DITHER_BAYER_PRECISION).r; }
vec3 ditherBayer(const in vec3 color, const in vec2 xy) {  return ditherBayer(color, xy, DITHER_BAYER_PRECISION); }
vec4 ditherBayer(const in vec4 color, const in vec2 xy) {  return vec4(ditherBayer(color.rgb, xy, DITHER_BAYER_PRECISION), color.a); }

float ditherBayer(const in float val, int pres) { return ditherBayer(vec3(val),DITHER_BAKER_COORD, pres).r; }
vec3 ditherBayer(const in vec3 color, int pres) { return ditherBayer(color, DITHER_BAKER_COORD, pres); }
vec4 ditherBayer(const in vec4 color, int pres) { return vec4(ditherBayer(color.rgb, DITHER_BAKER_COORD, pres), color.a); }

float ditherBayer(const in float val) { return ditherBayer(vec3(val), DITHER_BAKER_COORD, DITHER_BAYER_PRECISION).r; }
vec3 ditherBayer(const in vec3 color) { return ditherBayer(color, DITHER_BAKER_COORD, DITHER_BAYER_PRECISION); }
vec4 ditherBayer(const in vec4 color) { return vec4(ditherBayer(color.rgb), color.a); }
#endif