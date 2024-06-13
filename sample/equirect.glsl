#include "../space/xyz2equirect.glsl"
#include "../generative/random.glsl"
#include "../generative/srandom.glsl"
#include "../sampler.glsl"

#include "../color/space/linear2gamma.glsl"
#include "../color/space/gamma2linear.glsl"
/*
contributors: Patricio Gonzalez Vivo
description: sample an equirect texture as it was a cubemap
use: sampleEquirect(<SAMPLER_TYPE> texture, <vec3> dir)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEEQUIRECT_ITERATIONS: null
    - SAMPLEEQUIRECT_FLIP_Y
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEEQUIRECT
#define FNC_SAMPLEEQUIRECT
vec4 sampleEquirect(SAMPLER_TYPE tex, vec3 dir) { 
    vec2 st = xyz2equirect(dir);
    #ifdef SAMPLEEQUIRECT_FLIP_Y
    st.y = 1.0-st.y;
    #endif
    return SAMPLER_FNC(tex, st); 
}

vec4 sampleEquirect(SAMPLER_TYPE tex, vec3 dir, float lod) { 
    
    #if defined(SAMPLEEQUIRECT_ITERATIONS)
    vec4 color = vec4(0.0);
    vec2 st = xyz2equirect(dir);
    #ifdef SAMPLEEQUIRECT_FLIP_Y
        st.y = 1.0-st.y;
    #endif

    vec2 r = vec2(1.0+lod);
    const float f = 1.0 / (1.001 - 0.75);
    mat2 rot = mat2( cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE), 
                    -sin(GOLDEN_ANGLE), cos(GOLDEN_ANGLE));
    vec2 st2 = vec2( dot(st + st - r, vec2(.0002,-0.001)), 0.0 );

    float counter = 0.0;
    #ifdef PLATFORM_WEBGL
    for (float i = 0.0; i < float(SAMPLEEQUIRECT_ITERATIONS); i++) {
    #else
    for (float i = 0.0; i < float(SAMPLEEQUIRECT_ITERATIONS); i += 2.0/i) {
    #endif
        st2 *= rot;
        color += gamma2linear( SAMPLER_FNC(tex, st + st2 * i / vec2(r.x * 2.0, r.y))) * f;
        counter++;
    }
    return linear2gamma(color / counter);

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