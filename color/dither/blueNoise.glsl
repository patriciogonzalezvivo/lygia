#include "../../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
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

#ifndef HIGHP
#if defined(TARGET_MOBILE) && defined(GL_ES)
#define HIGHP highp
#else
#define HIGHP
#endif
#endif

#ifndef BLUENOISE_TEXTURE_RESOLUTION
#define BLUENOISE_TEXTURE_RESOLUTION vec2(1024.0)
#endif

#ifdef DITHER_ANIMATED
#define DITHER_BLUENOISE_ANIMATED
#endif

#ifdef DITHER_CHROMATIC
#define DITHER_BLUENOISE_CHROMATIC
#endif

#ifndef DITHER_BLUENOISE
#define DITHER_BLUENOISE

float remap_pdf_tri_unity( float v ) {
    v = v*2.0-1.0;
    v = sign(v) * (1.0 - sqrt(1.0 - abs(v)));
    return 0.5 + 0.5*v;
}

const vec2 blueNoiseTexturePixel = 1.0/BLUENOISE_TEXTURE_RESOLUTION;

float ditherBlueNoise(sampler2D tex, in float b, const HIGHP in float time) {
    vec2 st = gl_FragCoord.xy;
    #ifdef DITHER_BLUENOISE_ANIMATED 
    st += 1337.0*fract(time);
    #endif
    float bn = SAMPLER_FNC(tex, st * blueNoiseTexturePixel).r;
    float bn_tri = remap_pdf_tri_unity(bn);
    return b + (bn_tri*2.0-0.5)/255.0;
}

vec3 ditherBlueNoise(sampler2D tex, in vec3 rgb, const HIGHP in float time) {
    vec2 st = gl_FragCoord.xy;
    #ifdef DITHER_BLUENOISE_ANIMATED
    st += 1337.0*fract(time * 0.1);
    #endif
        
    #ifdef DITHER_BLUENOISE_CHROMATIC
    vec3 bn = SAM_PLER_FNC(tex, st * blueNoiseTexturePixel).rgb;
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

vec4 ditherBlueNoise(sampler2D tex, in vec4 rgba, const HIGHP in float time) {
    return vec4(ditherBlueNoise(tex, rgba.rgb, time), rgba.a);
}

#if defined(BLUENOISE_TEXTURE)
float ditherBlueNoise(const in float b, const HIGHP in float time) {
    return ditherBlueNoise(BLUENOISE_TEXTURE, b, time);
}

vec3 ditherBlueNoise(const in vec3 rgb, const HIGHP in float time) {
    return ditherBlueNoise(BLUENOISE_TEXTURE, rgb, time);
}

vec4 ditherBlueNoise(const in vec4 rgba, const HIGHP in float time) {
    return ditherBlueNoise(BLUENOISE_TEXTURE, rgba, time);
}
#endif

#endif