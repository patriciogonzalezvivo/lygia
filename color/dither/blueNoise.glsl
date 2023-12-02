#include "../../sample.glsl"
#include "../../math/decimate.glsl"
#include "../../math/saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: nan
use: 
 - <vec4|vec3|float> ditherBlueNoise(<vec4|vec3|float> value, <vec2> st, <float> time)
 - <vec4|vec3|float> ditherBlueNoise(<vec4|vec3|float> value, <float> time)
options:
    - SAMPLER_FNC
    - BLUENOISE_TEXTURE
    - BLUENOISE_TEXTURE_RESOLUTION
    - DITHER_BLUENOISE_CHROMATIC
    - DITHER_BLUENOISE_ANIMATED
examples:
    - /shaders/color_dither.frag
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

#ifndef BLUENOISE_TEXTURE_RESOLUTION
#define BLUENOISE_TEXTURE_RESOLUTION vec2(1024.0)
#endif

#ifndef DITHER_BLUENOISE
#define DITHER_BLUENOISE

float ditherBlueNoise(vec2 p) {
    const float SEED1 = 1.705;
    const float size = 5.5;
    vec2 p1 = p;
    p += 10.0;
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

vec3 ditherBlueNoise(vec3 color, const in vec2 xy, const int pres) {
    float d = float(pres);
    vec3 decimated = decimate(color, d);
    vec3 diff = (color - decimated) * d;
    color += step(vec3(ditherBlueNoise(xy)), diff) / d;
    color = decimate(color, d);
    return saturate(color);
}

float ditherBlueNoise(float val, const in vec2 xy) { return ditherBlueNoise(vec3(val), xy, 4).r; }
vec3 ditherBlueNoise(vec3 color, const in vec2 xy) { return ditherBlueNoise(color, xy, 4); }  
vec4 ditherBlueNoise(vec4 color, const in vec2 xy) {  return vec4(ditherBlueNoise(color.rgb, xy, 4), color.a); }

// float ditherBlueNoise(float val, int pres) { return ditherBlueNoise(vec3(val),DITHER_BLUENOISE_COORD, pres).r; }
// vec3 ditherBlueNoise(vec3 color, int pres) { return ditherBlueNoise(color, DITHER_BLUENOISE_COORD, pres); }
// vec4 ditherBlueNoise(vec4 color, int pres) { return vec4(ditherBlueNoise(color.rgb, DITHER_BLUENOISE_COORD, pres), color.a); }

float remap_pdf_tri_unity( float v ) {
    v = v*2.0-1.0;
    v = sign(v) * (1.0 - sqrt(1.0 - abs(v)));
    return 0.5 + 0.5*v;
}

const vec2 blueNoiseTexturePixel = 1.0/BLUENOISE_TEXTURE_RESOLUTION;

float ditherBlueNoise(SAMPLER_TYPE tex, in float b, vec2 st) {
    #ifdef DITHER_BLUENOISE_TIME 
    st += 1337.0*fract(DITHER_BLUENOISE_TIME);
    #endif
    float bn = SAMPLER_FNC(tex, st * blueNoiseTexturePixel).r;
    float bn_tri = remap_pdf_tri_unity(bn);
    return b + (bn_tri*2.0-0.5)/255.0;
}

vec3 ditherBlueNoise(SAMPLER_TYPE tex, in vec3 rgb, vec2 st) {
    #ifdef DITHER_BLUENOISE_TIME
    st += 1337.0*fract(DITHER_BLUENOISE_TIME * 0.1);
    #endif
        
    #ifdef DITHER_BLUENOISE_CHROMATIC
    vec3 bn = SAMPLER_FNC(tex, st * blueNoiseTexturePixel).rgb;
    vec3 bn_tri = vec3( remap_noise_tri_erp(bn.x), 
                        remap_noise_tri_erp(bn.y), 
                        remap_noise_tri_erp(bn.z) );
    rgb += (bn_tri*2.0-0.5)/255.0;
    #else
    float bn = SAMPLER_FNC(tex, st * blueNoiseTexturePixel).r;
    float bn_tri = remap_pdf_tri_unity(bn);
    rgb += (bn_tri*2.0-0.5)/255.0;
    #endif

    return rgb;
}

vec4 ditherBlueNoise(SAMPLER_TYPE tex, in vec4 rgba, vec2 st) { return vec4(ditherBlueNoise(tex, rgba.rgb, st), rgba.a); }

#ifdef BLUENOISE_TEXTURE
float ditherBlueNoise(float val) { return ditherBlueNoise(BLUENOISE_TEXTURE, val, DITHER_BLUENOISE_COORD); }
vec3 ditherBlueNoise(vec3 color) { return ditherBlueNoise(BLUENOISE_TEXTURE, color, DITHER_BLUENOISE_COORD); }
vec4 ditherBlueNoise(vec4 color) { return ditherBlueNoise(BLUENOISE_TEXTURE, color, DITHER_BLUENOISE_COORD); }
#else 
float ditherBlueNoise(float val) { return ditherBlueNoise(val, DITHER_BLUENOISE_COORD); }
vec3 ditherBlueNoise(vec3 color) { return ditherBlueNoise(color, DITHER_BLUENOISE_COORD); }
vec4 ditherBlueNoise(vec4 color) { return ditherBlueNoise(color, DITHER_BLUENOISE_COORD); }
#endif

#endif