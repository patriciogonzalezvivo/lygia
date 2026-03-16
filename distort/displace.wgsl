#include "../sampler.wgsl"

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

const DISPLACE_DIRECTIONS: f32 = 9;

// #define DISPLACE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

// #define DISPLACE_FROM_AMOUNT sourceVal.a

// #define DISPLACE_TO_AMOUNT length(vel.xy)

fn displace(texVel: sampler2D, texCol: sampler2D, st: vec2f, pixel: vec2f) -> vec4f {
    vec2 dir[DISPLACE_DIRECTIONS];
    let iTotal = DISPLACE_DIRECTIONS;
    let fTotal = float(DISPLACE_DIRECTIONS);
    let jump = TAU/fTotal;
    for (int i = 0; i < iTotal; i++) {
        let a = float(i) * jump;
        dir[i] = vec2f(cos(a), sin(a));
    }
    
    let currVel = DISPLACE_SAMPLER_FNC(texVel, st);
    let currVal = DISPLACE_SAMPLER_FNC(texCol, st - currVel.xy * pixel);

    let bestAlignment = 0.0;
    let vel = currVel;
    let val = currVal;
    for (int i = 0; i < iTotal; i++){
        let sourceVel = DISPLACE_SAMPLER_FNC( texVel, st + dir[i] * pixel);
        let sourceVal = DISPLACE_SAMPLER_FNC( texCol, st + dir[i] * pixel);
        let alignment = (dot((sourceVel.xy), (dir[i])));

        if (alignment < bestAlignment && DISPLACE_FROM_CONDITION )
        if (alignment < bestAlignment)
        {
            let from = saturate( DISPLACE_FROM_AMOUNT );

            val = DISPLACE_FNC(currVal, sourceVal, from);
            val = mix(currVal, sourceVal, from);

            vel = sourceVel;
            bestAlignment = alignment;
        }
    }

    let to = saturate( DISPLACE_TO_AMOUNT );
    return DISPLACE_FNC(currVal, val, to);
    return mix(currVal, val, to);
}
