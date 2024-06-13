#include "../math/const.hlsl"
#include "../generative/random.hlsl"
#include "../sampler.hlsl"

/*
contributors: Inigo Quiles
description: |
    Avoiding texture repetition by using Voronoise: a small texture can be used to generate infinite variety instead of tiled repetition. More info:  https://iquilezles.org/articles/texturerepetition/
use: sampleUNTILE(<SAMPLER_TYPE> texture, <float2> st, <float> noTiling)
options:
    - SAMPLER_FNC(TEX, UV)
    - SAMPLEUNTILE_TYPE
    - SAMPLEUNTILE_SAMPLER_FNC(UV)
*/

#ifndef SAMPLEUNTILE_TYPE
#define SAMPLEUNTILE_TYPE float4
#endif

#ifndef SAMPLEUNTILE_SAMPLER_FNC
#define SAMPLEUNTILE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef SAMPLEUNTILE_RANDOM_FNC 
#define SAMPLEUNTILE_RANDOM_FNC(XYZ) random4(XYZ) 
#endif

#ifndef FNC_SAMPLEUNTILE
#define FNC_SAMPLEUNTILE

SAMPLEUNTILE_TYPE sampleUntile(SAMPLER_TYPE tex, in float2 st, float v) {
    float2 p = floor( st );
    float2 f = frac( st );
        
    // derivatives (for correct mipmapping)
    // float2 ddx = dFdx( st );
    // float2 ddy = dFdy( st );
    
    SAMPLEUNTILE_TYPE va = float4(0.0, 0.0, 0.0, 0.0);
    float w1 = 0.0;
    float w2 = 0.0;
    float2 g = float2(-1.0, -1.0);
    for( g.y = -1.0; g.y <= 1.0; g.y++ )
    for( g.x = -1.0; g.x <= 1.0; g.x++ ) {
        float4 o = SAMPLEUNTILE_RANDOM_FNC( p + g );
        float2 r = g - f + o.xy;
        float d = dot(r,r);
        float w = exp(-5.0*d );
        // SAMPLEUNTILE_TYPE c = textureGrad(tex, st + v*o.zw, ddx, ddy );
        SAMPLEUNTILE_TYPE c = SAMPLEUNTILE_SAMPLER_FNC(tex, st + v*o.zw); 
        va += w*c;
        w1 += w;
        w2 += w*w;
    }
    
    // normal averaging --> lowers contrasts
    // return va/w1;

    // contrast preserving average
    float mean = 0.3;// textureGrad( samp, uv, ddx*16.0, ddy*16.0 ).x;
    SAMPLEUNTILE_TYPE res = mean + (va-w1*mean)/sqrt(w2);
    return lerp( va/w1, res, v );
}

#endif