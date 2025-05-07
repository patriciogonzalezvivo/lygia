/*
author: John Hable
description: Tonemapping function from presentation, uncharted 2 HDR Lighting, Page 142 to 143
use: <vec3|vec4> tonemapUncharted2(<vec3|vec4> x)
*/

fn tonemapUncharted23(v : vec3f) -> vec3f {
    let A = 0.15;  // 0.22
    let B = 0.50;  // 0.30
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;  // 0.01
    let F = 0.30;
    let W = 11.2;

    var x = vec4(v, W);
    x = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    return x.xyz / x.w;
}

fn tonemapUncharted24(x : vec4f) -> vec4f { return vec4(tonemapUncharted23(x.rgb), x.a); }
