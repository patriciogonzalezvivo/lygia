
/*
contributors: Johan Ismael
description: Sharpening convolutional operation
use:
    - sharpenFast4(texture, sampler, coords, pixel, strength) -> vec4f
    - sharpenFast(texture, sampler, coords, pixel) -> vec4f
options:
    - SHARPENFAST_KERNELSIZE: const i32, defaults to 2 (line 13)
*/

const SHARPENFAST_KERNELSIZE = 2;

fn sharpenFast4(myTexture: texture_2d<f32>,
                mySampler: sampler,
                coords: vec2f,
                pixel: vec2f,
                strength: f32) -> vec4f {
    var sum = vec4f(0.0);
    for (var i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        var f_size = f32(i) + 1.0;
        f_size *= strength;
        sum += -1.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(-1.0, 0.0) * pixel * f_size);
        sum += -1.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0.0, -1.0) * pixel * f_size);
        sum += 5.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0.0, 0.0) * pixel * f_size);
        sum += -1.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0.0, 1.0) * pixel * f_size);
        sum += -1.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(1.0, 0.0) * pixel * f_size);
    }
    return sum / f32(SHARPENFAST_KERNELSIZE);
}

fn sharpenFast(myTexture: texture_2d<f32>,
               mySampler: sampler,
               coords: vec2f,
               pixel: vec2f) -> vec4f {
    var sum = vec4f(0.0);
    for (var i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        let f_size = f32(i) + 1.0;
        sum += -1.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(-1.0, 0.0) * pixel * f_size);
        sum += -1.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0.0, -1.0) * pixel * f_size);
        sum += 5.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0.0, 0.0) * pixel * f_size);
        sum += -1.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(0.0, 1.0) * pixel * f_size);
        sum += -1.0 * textureSampleBaseClampToEdge(myTexture, mySampler, coords + vec2(1.0, 0.0) * pixel * f_size);
    }
    return sum / f32(SHARPENFAST_KERNELSIZE);
}
