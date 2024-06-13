#include "../generative/snoise.glsl"
#include "../generative/pnoise.glsl"
#include "../color/luma.glsl"
#include "../color/blend/softLight.glsl"
#include "../sampler.glsl"

/*
contributors: Matt DesLauriers
description: Natural looking film grain using 3D noise functions (original source https://github.com/mattdesl/glsl-film-grain). Inspired by [Martins Upitis](http://devlog-martinsh.blogspot.com/2013/05/image-imperfections-and-film-grain-post.html).
use: 
    - grain(<vec2> texCoord, <vec2> resolution [, <float> t, <float> multiplier])
    - grain(<SAMPLER_TYPE> texture, <vec2> texCoord, <float|vec2> resolution [, <float> t, <float> multiplier])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GRAIN_TYPE: type of the returned value (vec3 by default)
    - GRAIN_SAMPLER_FNC: grain function for SAMPLER_TYPE
license: MIT License (MIT) Copyright (c) 2015 Matt DesLauriers
*/

#ifndef GRAIN_TYPE
#define GRAIN_TYPE vec3
#endif

#ifndef GRAIN_SAMPLER_FNC
#define GRAIN_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb
#endif

#ifndef FNC_GRAIN
#define FNC_GRAIN
float grain(vec2 texCoord, vec2 resolution, float t, float multiplier) {
    vec2 mult = texCoord * resolution;
    float offset = snoise(vec3(mult / multiplier, t));
    float n1 = pnoise(vec3(mult, offset), vec3(1. / texCoord * resolution, 1.));
    return n1 / 2. + .5;
}

float grain(vec2 texCoord, vec2 resolution, float t) {
    return grain(texCoord, resolution, t, 2.5);
}

float grain(vec2 texCoord, vec2 resolution) {
    return grain(texCoord, resolution, 0.);
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, vec2 st, vec2 resolution, float t, float multiplier ) {
    GRAIN_TYPE org = GRAIN_SAMPLER_FNC(tex, st);

    float g = grain(st, resolution, t, multiplier);

    //get the luminance of the background
    float luminance = luma(org);
    
    //reduce the noise based on some 
    //threshold of the background luminance
    float response = smoothstep(0.05, 0.5, luminance);
    return mix( blendSoftLight(org, GRAIN_TYPE(g)), 
                org, 
                response * response);
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, vec2 st, vec2 resolution, float t ) {
    return grain(tex, st, resolution, t, 2.5 );
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, vec2 st, vec2 resolution) {
    return grain(tex, st, resolution, 0.);
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, vec2 st, float resolution, float t, float multiplier  ) {
    return grain(tex, st, vec2(resolution), t, multiplier );
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, vec2 st, float resolution, float t ) {
    return grain(tex, st, resolution, t, 2.5 );
}

GRAIN_TYPE grain(SAMPLER_TYPE tex, vec2 st, float resolution) {
    return grain(tex, st, resolution, 0.);
}

#endif