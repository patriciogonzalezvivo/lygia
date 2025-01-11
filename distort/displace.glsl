
#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Displace pixels
use: <vec4> displace(<sampler2D> texVel, <sampler2D> texCol, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - DISPLACE_SAMPLER_FNC: function used to sample the input texture, defaults to texture2D(TEX, UV).rgb
    - DISPLACE_FROM_CONDITION: condition to use the source of the mix
    - DISPLACE_FROM_AMOUNT: amount of the source to use
    - DISPLACE_TO_AMOUNT: amount of the target to use
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DISPLACE_DIRECTIONS
#define DISPLACE_DIRECTIONS 9
#endif

#ifndef DISPLACE_SAMPLER_FNC
#define DISPLACE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef DISPLACE_FROM_AMOUNT 
#define DISPLACE_FROM_AMOUNT sourceVal.a
#endif

#ifndef DISPLACE_TO_AMOUNT 
#define DISPLACE_TO_AMOUNT length(vel.xy)
#endif

#ifndef FNC_DISPLACE
#define FNC_DISPLACE

vec4 displace(sampler2D texVel, sampler2D texCol, vec2 st, vec2 pixel) {
    vec2 dir[DISPLACE_DIRECTIONS];
    int iTotal = DISPLACE_DIRECTIONS;
    float fTotal = float(DISPLACE_DIRECTIONS);
    float jump = TAU/fTotal;
    for (int i = 0; i < iTotal; i++) {
        float a = float(i) * jump;
        dir[i] = vec2(cos(a), sin(a));
    }
    
    vec4 currVel = DISPLACE_SAMPLER_FNC(texVel, st);
    vec4 currVal = DISPLACE_SAMPLER_FNC(texCol, st - currVel.xy * pixel);

    float bestAlignment = 0.0;
    vec4 vel = currVel;
    vec4 val = currVal;
    for (int i = 0; i < iTotal; i++){
        vec4 sourceVel = DISPLACE_SAMPLER_FNC( texVel, st + dir[i] * pixel);
        vec4 sourceVal = DISPLACE_SAMPLER_FNC( texCol, st + dir[i] * pixel);
        float alignment = (dot((sourceVel.xy), (dir[i])));

        #if defined(DISPLACE_FROM_CONDITION)
        if (alignment < bestAlignment && DISPLACE_FROM_CONDITION )
        #else 
        if (alignment < bestAlignment)
        #endif
        {
            float from = saturate( DISPLACE_FROM_AMOUNT );

            #if defined(DISPLACE_FNC)
            val = DISPLACE_FNC(currVal, sourceVal, from);
            #else
            val = mix(currVal, sourceVal, from);
            #endif

            vel = sourceVel;
            bestAlignment = alignment;
        }
    }

    float to = saturate( DISPLACE_TO_AMOUNT );
    #if defined(DISPLACE_FNC)
    return DISPLACE_FNC(currVal, val, to);
    #else
    return mix(currVal, val, to);
    #endif
}
#endif