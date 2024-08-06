#include "random.wgsl"
#include "../math/rotate2d.wgsl"

/*
contributors: Martijn Steinrucken
description: Wavelet noise https://www.shadertoy.com/view/wsBfzK
use: <vec2> worley(<vec2|vec3> pos)
options:
    - WAVELET_VORTICITY: amount of vorticity, i.e. spinning behaviour. With 0.0 (none) being the default, values may exceed 1.0.
examples:
    - /shaders/generative_worley.frag
license:
    - The MIT License Copyright 2020 Martijn Steinrucken
*/

const WAVELET_VORTICITY: f32 = 0.0;

fn wavelet(p: vec2f, phase: f32, scale: f32) -> f32 {
    var d = 0.0; 
    var s = 1.0;
    var m = 0.0;
    var a = 0.0;
    var tmp = p;
    for (var i = 0.0; i < 4.0; i += 1.0) {
        var q = tmp*s;
        a = random2(floor(q)) * 1e3;
        a += phase * random2(floor(q)) * WAVELET_VORTICITY;
        q = (fract(q) - 0.5) * rotate2d(a);
        d += sin(q.x * 10.0 + phase) * smoothstep(.25, 0.0, dot(q,q)) / s;
        tmp = tmp * mat2x2(0.54,-0.84, 0.84, 0.54) + i;
        m += 1.0 / s;
        s *= scale; 
    }
    return d / m;
}

fn waveletScaled3(p: vec3f, scale: f32) -> f32 {
    return wavelet(p.xy, p.z, scale);
}

fn wavelet3(p: vec3f) -> f32 {
    return wavelet(p.xy, p.z, 1.24);
} 

fn waveletScaled2(p: vec2f, phase: f32) -> f32 {
    return wavelet(p, phase, 1.24);
} 

fn wavelet2(p: vec2f) -> f32 {
    return wavelet(p, 0.0, 1.24);
} 
