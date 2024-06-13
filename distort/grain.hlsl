#include "../generative/snoise.hlsl"
#include "../generative/pnoise.hlsl"
#include "../color/luma.hlsl"
#include "../color/blend/softLight.hlsl"
#include "../sampler.hlsl"

/*
contributors: Matt DesLauriers
description: Natural looking film grain using 3D noise functions (original source https://github.com/mattdesl/glsl-film-grain). Inspired by [Martins Upitis](http://devlog-martinsh.blogspot.com/2013/05/image-imperfections-and-film-grain-post.html).
use: 
    - grain(<float2> texCoord, <float2> resolution [, <float> t, <float> multiplier])
    - grain(<SAMPLER_TYPE> texture, <float2> texCoord, <float|float2> resolution [, <float> t, <float> multiplier])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GRAIN_TYPE: type of the returned value (float3 by default)
    - GRAIN_SAMPLER_FNC: grain function for SAMPLER_TYPE
license: MIT License (MIT) Copyright (c) 2015 Matt DesLauriers
*/

#ifndef GRAIN_TYPE
#define GRAIN_TYPE float3
#endif

#ifndef GRAIN_SAMPLER_FNC
#define GRAIN_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb
#endif

#ifndef FNC_GRAIN
#define FNC_GRAIN
float grain(float2 texCoord, float2 resolution, float t, float multiplier) {
    float2 mult = texCoord * resolution;
    float offset = snoise(float3(mult / multiplier, t));
    float n1 = pnoise(float3(mult, offset), float3(1. / texCoord * resolution, 1.));
    return n1 / 2. + .5;
}

float grain(float2 texCoord, float2 resolution, float t) {
    return grain(texCoord, resolution, t, 2.5);
}

float grain(float2 texCoord, float2 resolution) {
    return grain(texCoord, resolution, 0.);
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, float2 st, float2 resolution, float t, float multiplier ) {
    GRAIN_TYPE org = GRAIN_SAMPLER_FNC(tex, st);

    float g = grain(st, resolution, t, multiplier);

    //get the luminance of the background
    float luminance = luma(org);
    
    //reduce the noise based on some 
    //threshold of the background luminance
    float response = smoothstep(0.05, 0.5, luminance);
    return lerp(   blendSoftLight(org, float4(g, g, g, 1.)), 
                    org, 
                    response * response);
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, float2 st, float2 resolution, float t ) {
    return grain(tex, st, resolution, t, 2.5 );
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, float2 st, float2 resolution) {
    return grain(tex, st, resolution, 0.);
}

#endif
