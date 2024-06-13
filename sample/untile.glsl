#include "../math/const.glsl"
#include "../math/sum.glsl"
#include "../generative/random.glsl"
#include "../sampler.glsl"

/*
contributors: Inigo Quiles
description: |
    Avoiding texture repetition by using Voronoise: a small texture can be used to generate infinite variety instead of tiled repetition. More info:  https://iquilezles.org/articles/texturerepetition/
use: sampleUNTILE(<SAMPLER_TYPE> texture, <vec2> st, <float> noTiling)
options:
    - SAMPLER_FNC(TEX, UV)
    - SAMPLEUNTILE_TYPE
    - SAMPLEUNTILE_SAMPLER_FNC(UV)
examples:
    - /shaders/sample_wrap_untile.frag
*/

#ifndef SAMPLEUNTILE_TYPE
#define SAMPLEUNTILE_TYPE vec4
#endif

#ifdef GL_OES_standard_derivatives
#extension GL_OES_standard_derivatives : enable
#endif

#ifndef SAMPLEUNTILE_SAMPLER_FNC
#if defined(PLATFORM_WEBGL) && __VERSION__ >= 300 && defined(GL_OES_standard_derivatives)
#define SAMPLEUNTILE_SAMPLER_FNC(TEX, UV) textureGrad(TEX, UV, ddx, ddy)
#else
#define SAMPLEUNTILE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef SAMPLEUNTILE_RANDOM_FNC 
#define SAMPLEUNTILE_RANDOM_FNC(XYZ) random4(XYZ) 
#endif

#ifndef FNC_SAMPLEUNTILE
#define FNC_SAMPLEUNTILE

SAMPLEUNTILE_TYPE sampleUntile(SAMPLER_TYPE tex, in vec2 st) {
        
    #if defined(PLATFORM_WEBGL) && __VERSION__ >= 300 && defined(GL_OES_standard_derivatives)
    vec2 ddx = dFdx( st );
    vec2 ddy = dFdy( st );
    #endif

    #ifdef SAMPLEUNTILE_FAST
    float k = SAMPLEUNTILE_SAMPLER_FNC(tex, 0.005*st ).x; // cheap (cache friendly) lookup
    
    float l = k*8.0;
    float f = fract(l);
    
    #if 0
    float ia = floor(l); // IQ method
    float ib = ia + 1.0;
    #else
    float ia = floor(l+0.5); // suslik's method
    float ib = floor(l);
    f = min(f, 1.0-f)*2.0;
    #endif    
    
    vec2 offa = sin(vec2(3.0,7.0) * ia); // can replace with any other hash
    vec2 offb = sin(vec2(3.0,7.0) * ib); // can replace with any other hash

    SAMPLEUNTILE_TYPE cola = SAMPLEUNTILE_SAMPLER_FNC(tex, st + offa );
    SAMPLEUNTILE_TYPE colb = SAMPLEUNTILE_SAMPLER_FNC(tex, st + offb );
    return mix( cola, colb, smoothstep(0.2, 0.8, f - 0.1 * sum(cola-colb) ) );

    #else 

    // More expensive because it samples x9
    // 
    vec2 p = floor( st );
    vec2 f = fract( st );
    
    SAMPLEUNTILE_TYPE va = SAMPLEUNTILE_TYPE(0.0);
    float w1 = 0.0;
    float w2 = 0.0;
    for( float y = -1.0; y <= 1.0; y++ )
    for( float x = -1.0; x <= 1.0; x++ ) {
        vec2 g = vec2(x, y);
        vec4 o = SAMPLEUNTILE_RANDOM_FNC( p + g );
        vec2 r = g - f + o.xy;
        float d = dot(r,r);
        float w = exp(-5.0*d );
        SAMPLEUNTILE_TYPE c = SAMPLEUNTILE_SAMPLER_FNC(tex, st + o.zw); 
        va += w*c;
        w1 += w;
        w2 += w*w;
    }
    
    // normal averaging --> lowers contrasts
    // return va/w1;

    // contrast preserving average
    float mean = 0.3;
    SAMPLEUNTILE_TYPE res = mean + (va-w1*mean)/sqrt(w2);
    return res;
    #endif
}

#endif