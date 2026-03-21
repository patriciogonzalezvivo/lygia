#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample a texture with a looping flow animation, using a direction to push, an elapse time and a cycle.
use: sampleFlow(<SAMPLER_TYPE> tex, <vec2> st, <vec2> dir, <float> time, <float> cycle)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleFlow(tex: SAMPLER_TYPE, st: vec2f, dir: vec2f, time: f32, cycle: f32) -> vec4f {
    let halfCycle = cycle * 0.5;

    let flowOffset0 = mod(time, cycle);
    let flowOffset1 = mod(time + halfCycle, cycle);

    let phase0 = flowOffset0;
    let phase1 = flowOffset1;

    // Sample normal map.
    let A = SAMPLER_FNC(tex, (st + dir * phase0) );
    let B = SAMPLER_FNC(tex, (st + dir * phase1) );

    let f = (abs(halfCycle - flowOffset0) / halfCycle);
    return mix( A, B, f );
}
