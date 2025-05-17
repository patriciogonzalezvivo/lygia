import lygia::color::space::rgb2luma;

/*
contributors: Christian Cann Schuldt Jensen ~ CeeJay.dk
description: |
    Vibrance is a smart-tool which cleverly increases the intensity of the more muted colors and leaves the already well-saturated colors alone. Prevents skin tones from becoming overly saturated and unnatural.
    vibrance from https://github.com/CeeJayDK/SweetFX/blob/master/Shaders/Vibrance.fx
use: <vec3|vec4> vibrance(<vec3|vec4> color, <float> v)
license: MIT License (MIT) Copyright (c) 2014 CeeJayDK
*/

fn vibrance3(color : vec3f, v : f32) -> vec3f {
    let max_color = max(color.r, max(color.g, color.b));
    let min_color = min(color.r, min(color.g, color.b));
    let sat = max_color - min_color;
    return mix(vec3f(rgb2luma(color)), color, 1.0 + (v * 1.0 - (sign(v) * sat)));
}

fn vibrance4(color : vec4f, v : f32) -> vec4f { return vec4(vibrance3(color.rgb, v), color.a); }