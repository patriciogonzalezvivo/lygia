import lygia::color::space::rgb2oklab::RGB2OKLAB_B;
import lygia::color::space::oklab2rgb::OKLAB2RGB_B;
import lygia::color::space::{srgb2rgb::srgb2rgb, rgb2srgb::rgb2srgb};

/*
contributors:
    - Bjorn Ottosson
    - Inigo Quiles
description: |
    Mix function by Inigo Quiles (https://www.shadertoy.com/view/ttcyRS)
    utilizing Bjorn Ottosson's OkLab color space, which provides smooth transitions.
    Learn more about it in [his article](https://bottosson.github.io/posts/oklab/)
use: <vec3f|vec4f> mixOklab(<vec3f|vec4f> colorA, <vec3f|vec4f> colorB, <f32> pct)
note: |
    Two modes are provided via WESL conditionals:
    - @if(!MIXOKLAB_SRGB): Linear RGB input/output (default)
    - @if(MIXOKLAB_SRGB): sRGB input/output - performs conversion before/after mixing
examples:
    - /shaders/color_mix.frag
license:
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
    - MIT License (MIT) Copyright (c) 2020 Inigo Quilez
*/

fn mixOklab(colA: vec3f, colB: vec3f, h: f32) -> vec3f {
    @if(MIXOKLAB_SRGB)
    let colorA = srgb2rgb(colA);
    @else
    let colorA = colA;

    @if(MIXOKLAB_SRGB)
    let colorB = srgb2rgb(colB);
    @else
    let colorB = colB;

    // rgb to cone (arg of pow can't be negative)
    let lmsA = pow(RGB2OKLAB_B * colorA, vec3f(0.33333));
    let lmsB = pow(RGB2OKLAB_B * colorB, vec3f(0.33333));

    let lms = mix(lmsA, lmsB, h);

    // cone to rgb
    let rgb = OKLAB2RGB_B * (lms * lms * lms);

    @if(MIXOKLAB_SRGB)
    return rgb2srgb(rgb);
    @else
    return rgb;
}

fn mixOklab4(colA: vec4f, colB: vec4f, h: f32) -> vec4f {
    return vec4f(mixOklab(colA.rgb, colB.rgb, h), mix(colA.a, colB.a, h));
}
