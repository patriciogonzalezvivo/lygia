
/*
contributors: Johan Ismael
description: Sharpening convolutional operation
use: sharpen(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SHARPENFAST_KERNELSIZE: Defaults 2
    - SHARPENFAST_TYPE: defaults to vec3
    - SHARPENFAST_SAMPLER_FNC(TEX, UV): defaults to texture2D(tex, TEX, UV).rgb
*/

const SHARPENFAST_KERNELSIZE = 2;

fn sharpenFast(myTexture: texture_2d<f32>,
               mySampler: sampler,
               coords: vec2f,
               pixel: vec2f,
               strength: f32) -> vec4f {
    var sum = vec4f(0.);
    for (var i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        var f_size = f32(i) + 1.;
        f_size *= strength;
        sum += -1. * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(-1., 0.) * pixel * f_size);
        sum += -1. * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0., -1.) * pixel * f_size);
        sum += 5. * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0., 0.) * pixel * f_size);
        sum += -1. * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0., 1.) * pixel * f_size);
        sum += -1. * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(1., 0.) * pixel * f_size);
    }
    return sum / f32(SHARPENFAST_KERNELSIZE);
}
