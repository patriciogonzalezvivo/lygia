#include "../../math/saturate.glsl"
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
*/

#ifndef DITHER_BAKER_COORD
#define DITHER_BAKER_COORD gl_FragCoord.xy
#endif

#ifndef DITHER_BAYER_PRECISION
#ifdef DITHER_PRECISION
#define DITHER_BAYER_PRECISION DITHER_PRECISION
#else
#define DITHER_BAYER_PRECISION 255
#endif
#endif

#ifndef FNC_DITHER_BAYER
#define FNC_DITHER_BAYER

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

vec3 ditherBayer(const in vec3 color, const in vec2 xy, const int pres) {
    float d = float(pres);
    vec3 decimated = decimate(color, d);
    vec3 diff = (color - decimated) * d;
    vec3 ditherPattern = vec3(ditherBayer(xy));
    return saturate(decimate(color + (step(ditherPattern, diff) / d), d));
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