#include "../space/xyz2equirect.glsl"
#include "../generative/random.glsl"
#include "../generative/srandom.glsl"
#include "../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: sample an equirect texture as it was a cubemap
use: sampleEquirect(<sampler2D> texture, <vec3> dir)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEEQUIRET_ITERATIONS:
    - SAMPLEEQUIRECT_FLIP_Y
*/

#ifndef FNC_SAMPLEEQUIRECT
#define FNC_SAMPLEEQUIRECT
vec4 sampleEquirect(sampler2D tex, vec3 dir) { 
    vec2 st = xyz2equirect(dir);
    #ifdef SAMPLEEQUIRECT_FLIP_Y
    st.y = 1.0-st.y;
    #endif
    return SAMPLER_FNC(tex, st); 
}

vec4 sampleEquirect(sampler2D tex, vec3 dir, float lod) { 
    
    #if defined(SAMPLEEQUIRET_ITERATIONS)
    vec4 acc = vec4(0.0);
    vec2 st = xyz2equirect(dir);
    #ifdef SAMPLEEQUIRECT_FLIP_Y
    st.y = 1.0-st.y;
    #endif
    mat2 rot = mat2(cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE), -sin(GOLDEN_ANGLE), cos(GOLDEN_ANGLE));
    float r = 1.;
    vec2 vangle = vec2(0.0, lod * 0.01);
    for (int i = 0; i < SAMPLEEQUIRET_ITERATIONS; i++) {
        vangle = rot * vangle;
        r++;
        vec4 col = SAMPLER_FNC(tex, st + random( vec3(st, r) ) * vangle );
        acc += col * col;
    }
    return vec4(acc.rgb/acc.a, 1.0); 

    #else
    dir += srandom3( dir ) * 0.01 * lod;
    vec2 st = xyz2equirect(dir);
    #ifdef SAMPLEEQUIRECT_FLIP_Y
    st.y = 1.0-st.y;
    #endif
    return SAMPLER_FNC(tex, st);

    #endif
}

#endif