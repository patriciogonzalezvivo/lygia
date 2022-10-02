#include "../math/const.glsl"
#include "../generative/random.glsl"

/*
original_author: Inigo Quiles
description: Avoiding texture repetition by using Voronoise: a small texture can be used to generate infinite variety instead of tiled repetition. More info:  https://iquilezles.org/articles/texturerepetition/
use: sampleSeamless(<sampler2D> texture, <vec2> st, <float> noTiling)
options:
    - SAMPLER_FNC(TEX, UV)
    - SAMPLESEAMLESS_TYPE
    - SAMPLESEAMLESS_SAMPLER_FNC(UV)
*/
#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef SAMPLESEAMLESS_TYPE
#define SAMPLESEAMLESS_TYPE vec4
#endif

#ifndef SAMPLESEAMLESS_SAMPLER_FNC
#define SAMPLESEAMLESS_SAMPLER_FNC(UV) SAMPLER_FNC(tex, UV)
#endif

#ifndef SAMPLESEAMLESS_RANDOM_FNC 
#define SAMPLESEAMLESS_RANDOM_FNC(XYZ) random4(XYZ) 
#endif

#ifndef FNC_SAMPLESEAMLESS
#define FNC_SAMPLESEAMLESS

SAMPLESEAMLESS_TYPE sampleSeamless(sampler2D tex, in vec2 st, float v) {
    vec2 p = floor( st );
    vec2 f = fract( st );
        
    // derivatives (for correct mipmapping)
    // vec2 ddx = dFdx( st );
    // vec2 ddy = dFdy( st );
    
    SAMPLESEAMLESS_TYPE va = SAMPLESEAMLESS_TYPE(0.0);
    float w1 = 0.0;
    float w2 = 0.0;
    vec2 g = vec2(-1.0, -1.0);
    for( g.y = -1.0; g.y <= 1.0; g.y++ )
    for( g.x = -1.0; g.x <= 1.0; g.x++ ) {
        vec4 o = SAMPLESEAMLESS_RANDOM_FNC( p + g );
        vec2 r = g - f + o.xy;
        float d = dot(r,r);
        float w = exp(-5.0*d );
        // SAMPLESEAMLESS_TYPE c = textureGrad(tex, st + v*o.zw, ddx, ddy );
        SAMPLESEAMLESS_TYPE c = SAMPLESEAMLESS_SAMPLER_FNC(st + v*o.zw); 
        va += w*c;
        w1 += w;
        w2 += w*w;
    }
    
    // normal averaging --> lowers contrasts
    // return va/w1;

    // contrast preserving average
    float mean = 0.3;// textureGrad( samp, uv, ddx*16.0, ddy*16.0 ).x;
    SAMPLESEAMLESS_TYPE res = mean + (va-w1*mean)/sqrt(w2);
    return mix( va/w1, res, v );
}

#endif