#include "space/srgb2rgb.glsl"
#include "space/rgb2srgb.glsl"

#include "space/oklab2rgb.glsl"
#include "space/rgb2oklab.glsl"

/*
contributors: [Bjorn Ottosson, Inigo Quiles]
description: |
    Mix function by Inigo Quiles (https://www.shadertoy.com/view/ttcyRS) 
    utilizing Bjorn Ottosso's OkLab color space, which is provide smooth stransitions 
    Learn more about it [his article](https://bottosson.github.io/posts/oklab/)
use: <vec3\vec4> mixOklab(<vec3|vec4> colorA, <vec3|vec4> colorB, float pct)
options:
    - MIXOKLAB_SRGB: color argument are in sRGB and returns sRGB
examples:
    - /shaders/color_mix.frag
license: 
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
    - MIT License (MIT) Copyright (c) 2020 Inigo Quilez
*/

#ifndef FNC_MIXOKLAB
#define FNC_MIXOKLAB
vec3 mixOklab( vec3 colA, vec3 colB, float h ) {

    #ifdef MIXOKLAB_SRGB
    colA = srgb2rgb(colA);
    colB = srgb2rgb(colB);
    #endif

    vec3 lmsA = pow( RGB2OKLAB_B * colA, vec3(0.33333) );
    vec3 lmsB = pow( RGB2OKLAB_B * colB, vec3(0.33333) );
    
    // lerp
    vec3 lms = mix( lmsA, lmsB, h );
    
    // cone to rgb
    vec3 rgb = OKLAB2RGB_B*(lms*lms*lms);

    #ifdef MIXOKLAB_SRGB
    return rgb2srgb(rgb);
    #else
    return rgb;
    #endif
}

vec4 mixOklab( vec4 colA, vec4 colB, float h ) {
    return vec4( mixOklab(colA.rgb, colB.rgb, h), mix(colA.a, colB.a, h) );
}
#endif