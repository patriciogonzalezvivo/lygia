#include "../math/saturate.wgsl"
#include "../sampler.wgsl"

/*
original_author: Patricio Gonzalez Vivo
description: Grayscott Reaction-Diffusion
use: <vec3> grayscott(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel, <float> src [, <float> diffU, <float> diffV, <float> f, <float> k ])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - GRAYSCOTT_ITERATIONS
*/

fn grayscott(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f, src: f32, diffU: f32, diffV: f32, f: f32, k: f32) -> vec3f {
    const GRAYSCOTT_ITERATIONS: f32 = 9;
    
    float kernel[9];
    kernel[0] = 0.707106781;
    kernel[1] = 1.0;
    kernel[2] = 0.707106781;
    kernel[3] = 1.0;
    kernel[4] = -6.82842712;
    kernel[5] = 1.0;
    kernel[6] = 0.707106781;
    kernel[7] = 1.0;
    kernel[8] = 0.707106781;

    vec2 offset[9];
    offset[0] = pixel * vec2f(-1.0,-1.0);
    offset[1] = pixel * vec2f( 0.0,-1.0);
    offset[2] = pixel * vec2f( 1.0,-1.0);

    offset[3] = pixel * vec2f(-1.0,0.0);
    offset[4] = pixel * vec2f( 0.0,0.0);
    offset[5] = pixel * vec2f( 1.0,0.0);

    offset[6] = pixel * vec2f(-1.0,1.0);
    offset[7] = pixel * vec2f( 0.0,1.0);
    offset[8] = pixel * vec2f( 1.0,1.0);

    let current = SAMPLER_FNC(tex, st).rb;

    let lap = vec2f(0.0);
    for (int i=0; i < GRAYSCOTT_ITERATIONS; i++){
        let tmp = SAMPLER_FNC(tex, st + offset[i]).rb;
        lap += tmp * kernel[i];
    }

    let F = f + src * 0.025 - 0.0005;
    let K = k + src * 0.025 - 0.0005;

    let u = current.r;
    let v = current.g + src * 0.5;

    let uvv = u * v * v;

    let du = diffU * lap.r - uvv + F * (1.0 - u);
    let dv = diffV * lap.g + uvv - (F + K) * v;

    u += du * 0.6;
    v += dv * 0.6;
    return vec3f(saturate(u), 1.0 - u/v, saturate(v));
}

fn grayscotta(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f, src: f32) -> vec3f {
    return grayscott(tex, st, pixel, src, 0.25, 0.05, 0.1, 0.063); 
}
