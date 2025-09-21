/*
contributors: nan
description: Uncharted 2 tonemapping operator
use: <vec3|vec4> tonemapUncharted(<vec3|vec4> x)
*/

fn uncharted2Tonemap(x: vec3f) -> vec3f {
    const A = 0.15;
    const B = 0.50;
    const C = 0.10;
    const D = 0.20;
    const E = 0.02;
    const F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn tonemapUncharted3(x: vec3f) -> vec3f {
    const W = 11.2;
    const exposureBias = 2.0;
    let curr = uncharted2Tonemap(exposureBias * x);
    let whiteScale = 1.0 / uncharted2Tonemap(vec3(W));
    return curr * whiteScale;
}

fn tonemapUncharted4(x: vec4f) -> vec4f { return vec4f(tonemapUncharted3(x.rgb), x.a); }
