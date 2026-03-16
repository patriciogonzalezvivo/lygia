#include "shadow.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample shadow map using PCF
use: <float> sampleShadowLerp(<SAMPLER_TYPE> depths, <vec2> size, <vec2> uv, <float> compare)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleShadowLerp(depths: SAMPLER_TYPE, size: vec2f, uv: vec2f, compare: f32) -> f32 {
    let texelSize = vec2f(1.0)/size;
    let f = fract(uv*size+0.5);
    let centroidUV = floor(uv*size+0.5)/size;
    let lb = sampleShadow(depths, centroidUV+texelSize*vec2f(0.0, 0.0), compare);
    let lt = sampleShadow(depths, centroidUV+texelSize*vec2f(0.0, 1.0), compare);
    let rb = sampleShadow(depths, centroidUV+texelSize*vec2f(1.0, 0.0), compare);
    let rt = sampleShadow(depths, centroidUV+texelSize*vec2f(1.0, 1.0), compare);
    let a = mix(lb, lt, f.y);
    let b = mix(rb, rt, f.y);
    let c = mix(a, b, f.x);
    return c;
}
