#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample derrivatives
use: sampleDerivative(<SAMPLER_TYPE> tex, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (tex2D(...) or texture(...))
    - USE_DERIVATIVES
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLERDERIVATIVE_FNC
#define SAMPLERDERIVATIVE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r
#endif

#ifndef FNC_SAMPLEDERIVATIVE
#define FNC_SAMPLEDERIVATIVE
vec2 sampleDerivative(in SAMPLER_TYPE tex, in vec2 st, vec2 pixel) { 
    float p = SAMPLERDERIVATIVE_FNC(tex, st); 

    #if defined(SAMPLEDERRIVATIVE_DD)
    return -vec2(dFdx(p), dFdy(p));

    #elif defined(SAMPLEDERRIVATIVE_FAST)
    float h1 = SAMPLERDERIVATIVE_FNC(tex, st + vec2(pixel.x,0.0));
    float v1 = SAMPLERDERIVATIVE_FNC(tex, st + vec2(0.0,pixel.y));
    return (p - vec2(h1, v1));

    #else
    float center      = SAMPLERDERIVATIVE_FNC(tex, st);
    float topLeft     = SAMPLERDERIVATIVE_FNC(tex, st - pixel);
    float left        = SAMPLERDERIVATIVE_FNC(tex, st - vec2(pixel.x, .0));
    float bottomLeft  = SAMPLERDERIVATIVE_FNC(tex, st + vec2(-pixel.x, pixel.y));
    float top         = SAMPLERDERIVATIVE_FNC(tex, st - vec2(.0, pixel.y));
    float bottom      = SAMPLERDERIVATIVE_FNC(tex, st + vec2(.0, pixel.y));
    float topRight    = SAMPLERDERIVATIVE_FNC(tex, st + vec2(pixel.x, -pixel.y));
    float right       = SAMPLERDERIVATIVE_FNC(tex, st + vec2(pixel.x, .0));
    float bottomRight = SAMPLERDERIVATIVE_FNC(tex, st + pixel);
    
    float dX = topRight + 2. * right + bottomRight - topLeft - 2. * left - bottomLeft;
    float dY = bottomLeft + 2. * bottom + bottomRight - topLeft - 2. * top - topRight;

    return vec2(dX, dY);
    #endif
}
#endif
