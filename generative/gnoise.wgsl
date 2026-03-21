#include "random.wgsl"
#include "srandom.wgsl"
#include "../math/cubic.wgsl"
#include "../math/quintic.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Gradient Noise
use: gnoise(<float> x)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn gnoise(x: f32) -> f32 {
    float i = floor(x);  // integer
    float f = fract(x);  // fraction
    return mix(random(i), random(i + 1.0), smoothstep(0.,1.,f)); 
}

fn gnoise2(st: vec2f) -> f32 {
    let i = floor(st);
    let f = fract(st);
    let a = random(i);
    let b = random(i + vec2f(1.0, 0.0));
    let c = random(i + vec2f(0.0, 1.0));
    let d = random(i + vec2f(1.0, 1.0));
    let u = cubic(f);
    return mix( a, b, u.x) +
                (c - a)* u.y * (1.0 - u.x) +
                (d - b) * u.x * u.y;
}

fn gnoise3(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = quintic(f);
    return -1.0 + 2.0 * mix( mix( mix( random(i + vec3f(0.0,0.0,0.0)), 
                                        random(i + vec3f(1.0,0.0,0.0)), u.x),
                                mix( random(i + vec3f(0.0,1.0,0.0)), 
                                        random(i + vec3f(1.0,1.0,0.0)), u.x), u.y),
                            mix( mix( random(i + vec3f(0.0,0.0,1.0)), 
                                        random(i + vec3f(1.0,0.0,1.0)), u.x),
                                mix( random(i + vec3f(0.0,1.0,1.0)), 
                                        random(i + vec3f(1.0,1.0,1.0)), u.x), u.y), u.z );
}

fn gnoise3a(p: vec3f, tileLength: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
            
    let u = quintic(f);
        
    return mix( mix( mix( dot( srandom3(i + vec3f(0.0,0.0,0.0), tileLength), f - vec3f(0.0,0.0,0.0)), 
                            dot( srandom3(i + vec3f(1.0,0.0,0.0), tileLength), f - vec3f(1.0,0.0,0.0)), u.x),
                    mix( dot( srandom3(i + vec3f(0.0,1.0,0.0), tileLength), f - vec3f(0.0,1.0,0.0)), 
                            dot( srandom3(i + vec3f(1.0,1.0,0.0), tileLength), f - vec3f(1.0,1.0,0.0)), u.x), u.y),
                mix( mix( dot( srandom3(i + vec3f(0.0,0.0,1.0), tileLength), f - vec3f(0.0,0.0,1.0)), 
                            dot( srandom3(i + vec3f(1.0,0.0,1.0), tileLength), f - vec3f(1.0,0.0,1.0)), u.x),
                    mix( dot( srandom3(i + vec3f(0.0,1.0,1.0), tileLength), f - vec3f(0.0,1.0,1.0)), 
                            dot( srandom3(i + vec3f(1.0,1.0,1.0), tileLength), f - vec3f(1.0,1.0,1.0)), u.x), u.y), u.z );
}

fn gnoise3(x: vec3f) -> vec3f {
    return vec3f(gnoise(x+vec3f(123.456, 0.567, 0.37)),
                gnoise(x+vec3f(0.11, 47.43, 19.17)),
                gnoise(x) );
}
