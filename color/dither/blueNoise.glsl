#include "../../sampler.glsl"
#include "../../math/decimate.glsl"
#include "../../math/saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: blue noise dithering
use:
    - <vec4|vec3|float> ditherBlueNoise(<vec4|vec3|float> value, <vec2> st, <float> time)
    - <vec4|vec3|float> ditherBlueNoise(<vec4|vec3|float> value, <float> time)
options:
    - SAMPLER_FNC
    - BLUENOISE_TEXTURE
    - BLUENOISE_TEXTURE_RESOLUTION
    - DITHER_BLUENOISE_CHROMATIC
    - DITHER_BLUENOISE_TIME
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DITHER_BLUENOISE_COORD
#define DITHER_BLUENOISE_COORD gl_FragCoord.xy
#endif

#ifdef DITHER_TIME
#define DITHER_BLUENOISE_TIME DITHER_TIME
#endif

#ifdef DITHER_CHROMATIC
#define DITHER_BLUENOISE_CHROMATIC
#endif

#ifndef DITHER_BLUENOISE_PRECISION
#ifdef DITHER_PRECISION
#define DITHER_BLUENOISE_PRECISION DITHER_PRECISION
#else
#define DITHER_BLUENOISE_PRECISION 256
#endif
#endif

#ifndef BLUENOISE_TEXTURE_RESOLUTION
#define BLUENOISE_TEXTURE_RESOLUTION vec2(1024.0)
#endif

#ifndef DITHER_BLUENOISE
#define DITHER_BLUENOISE

#ifdef BLUENOISE_TEXTURE

float remap_pdf_tri_unity(float v) {
    v = v*2.0-1.0;
    return 0.5 + 0.5 * sign(v) * (1.0 - sqrt(1.0 - abs(v)));
}

const vec2 blueNoiseTexturePixel = 1.0/BLUENOISE_TEXTURE_RESOLUTION;

float ditherBlueNoise(SAMPLER_TYPE tex, const in float value, vec2 st, int pres) {
    float d = float(pres);
    #ifdef DITHER_BLUENOISE_TIME 
    st += 1337.0*fract(DITHER_BLUENOISE_TIME);
    #endif
    float bn = SAMPLER_FNC(tex, st * blueNoiseTexturePixel).r;
    float bn_tri = remap_pdf_tri_unity(bn);
    return value + (bn_tri*2.0-0.5)/d;
}

vec3 ditherBlueNoise(SAMPLER_TYPE tex, vec3 color, vec2 st, int pres) {
    float d = float(pres);

    #ifdef DITHER_BLUENOISE_TIME
    st += 1337.0*fract(DITHER_BLUENOISE_TIME * 0.1);
    #endif
        
    #ifdef DITHER_BLUENOISE_CHROMATIC
    vec3 bn = SAMPLER_FNC(tex, st * blueNoiseTexturePixel).color;
    vec3 bn_tri = vec3( remap_noise_tri_erp(bn.x), 
                        remap_noise_tri_erp(bn.y), 
                        remap_noise_tri_erp(bn.z) );
    color += (bn_tri*2.0-1.5)/d;
    #else
    float bn = SAMPLER_FNC(tex, st * blueNoiseTexturePixel).r;
    float bn_tri = remap_pdf_tri_unity(bn);
    color += (bn_tri*2.0-1.5)/d;
    #endif

    return color;
}

float ditherBlueNoise(SAMPLER_TYPE tex, const in float b, vec2 st) { return ditherBlueNoise(tex, b, st, DITHER_BLUENOISE_PRECISION); }
vec3 ditherBlueNoise(SAMPLER_TYPE tex, const in vec3 rgb, vec2 st) { return ditherBlueNoise(tex, rgb, st, DITHER_BLUENOISE_PRECISION);}
vec4 ditherBlueNoise(SAMPLER_TYPE tex, const in vec4 rgba, vec2 st) { return vec4(ditherBlueNoise(tex, rgba.rgb, st), rgba.a); }

float ditherBlueNoise(const in float val) { return ditherBlueNoise(BLUENOISE_TEXTURE, val, DITHER_BLUENOISE_COORD); }
vec3 ditherBlueNoise(const in vec3 color) { return ditherBlueNoise(BLUENOISE_TEXTURE, color, DITHER_BLUENOISE_COORD); }
vec4 ditherBlueNoise(const in vec4 color) { return ditherBlueNoise(BLUENOISE_TEXTURE, color, DITHER_BLUENOISE_COORD); }

#else 

float ditherBlueNoise(vec2 p) {
    const float SEED1 = 1.705;
    const float size = 5.5;
    p = floor(p);
    vec2 p1 = p;
    #ifdef DITHER_BLUENOISE_TIME
    p += 1337.0*fract(DITHER_BLUENOISE_TIME * 0.1);
    #else
    p += 10.0;
    #endif
    p = floor(p/size)*size;
    p = fract(p * 0.1) + 1.0 + p * vec2(0.0002, 0.0003);
    float a = fract(1.0 / (0.000001 * p.x * p.y + 0.00001));
    a = fract(1.0 / (0.000001234 * a + 0.00001));
    float b = fract(1.0 / (0.000002 * (p.x * p.y + p.x) + 0.00001));
    b = fract(1.0 / (0.0000235*b + 0.00001));
    vec2 r = vec2(a, b) - 0.5;
    p1 += r * 8.12235325;
    return fract(p1.x * SEED1 + p1.y/(SEED1+0.15555));
}

vec3 ditherBlueNoise(const in vec3 color, const in vec2 xy, const int pres) {
    float d = float(pres);
    vec3 decimated = decimate(color, d);
    vec3 diff = (color - decimated) * d;
    return saturate(decimate(color + step(vec3(ditherBlueNoise(xy)), diff) / d, d));
}

float ditherBlueNoise(const in float val, const in vec2 xy, const int pres) { return ditherBlueNoise(vec3(val), xy, pres).r; }
vec4 ditherBlueNoise(const in vec4 color, const in vec2 xy, const int pres) { return vec4(ditherBlueNoise(color.rgb, xy, pres), color.a); }

float ditherBlueNoise(const in float val, const in vec2 xy) { return ditherBlueNoise(vec3(val), xy, DITHER_BLUENOISE_PRECISION).r; }
vec3 ditherBlueNoise(const in vec3 color, const in vec2 xy) { return ditherBlueNoise(color, xy, DITHER_BLUENOISE_PRECISION); }  
vec4 ditherBlueNoise(const in vec4 color, const in vec2 xy) {  return vec4(ditherBlueNoise(color.rgb, xy, DITHER_BLUENOISE_PRECISION), color.a); }

float ditherBlueNoise(float val) { return ditherBlueNoise(val, DITHER_BLUENOISE_COORD, DITHER_BLUENOISE_PRECISION); }
vec3 ditherBlueNoise(vec3 color) { return ditherBlueNoise(color, DITHER_BLUENOISE_COORD, DITHER_BLUENOISE_PRECISION); }
vec4 ditherBlueNoise(vec4 color) { return ditherBlueNoise(color, DITHER_BLUENOISE_COORD, DITHER_BLUENOISE_PRECISION); }
#endif

#endif