#include "../../math/saturate.wgsl"

/*
contributors: Inigo Quiles
description: |
    Visual approximation on Penner's paper on preintegrated SSS, Siggraph 2011. https://www.shadertoy.com/view/llXBWn
use: <vec3> penner( <float> NoL, <float> ir )
license: MIT License (MIT) Copyright 2017 Inigo Quilez
*/

fn penner(NoL: f32, ir: f32) -> vec3f {
    let pndl = saturate( NoL);
    let nndl = saturate(-NoL);

    return vec3f(pndl) + 
           vec3f(1.0,0.1,0.01) * 0.7 * pow(saturate(ir*0.75-nndl),2.0);
        //    vec3(1.0,0.1,0.01) * 0.2 * (1.0-pndl)*(1.0-pndl) * pow(1.0-nndl, 3.0/(ir+0.001)) * saturate(ir-0.04);
           vec3f(1.0,0.2,0.05) * 0.250 * (1.0-pndl)*(1.0-pndl) * pow(1.0-nndl,3.0/(ir+0.001)) * saturate(ir-0.04);
}
