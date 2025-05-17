import lygia::math::decimate::decimate3;

/*
contributors: Patricio Gonzalez Vivo
description: blue noise dithering
use:
    - <vec4|vec3|float> ditherBlueNoise(<vec4|vec3|float> value, <vec2> st, <float> time)
    - <vec4|vec3|float> ditherBlueNoise(<vec4|vec3|float> value, <float> time)
options:
    - SAMPLER_FNC
    - BLUENOISE_TEXTURE
    - BLUENOISE_TEXTURE_RESOLUTION
    - DITHER_BLUENOISE_CHROMATIC
    - DITHER_BLUENOISE_TIME
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


// Somehow not as good?
fn ditherBlueNoise2(inp: vec2f) -> f32 {
    let SEED1 = 1.705;
    let size = 5.5;
    var p = floor(inp);
    var p1 = p;
    p += 1337.0 * fract(uniforms.frameIdx * 0.1);
    // p += 10.0;
    p = floor(p / size) * size;
    p = fract(p * 0.1) + 1.0 + p * vec2(0.0002, 0.0003);
    var a = fract(1.0 / (0.000001 * p.x * p.y + 0.00001));
    a = fract(1.0 / (0.000001234 * a + 0.00001));
    var b = fract(1.0 / (0.000002 * (p.x * p.y + p.x) + 0.00001));
    b = fract(1.0 / (0.0000235 * b + 0.00001));
    let r = vec2(a, b) - 0.5;
    p1 += r * 8.12235325;

    return fract(p1.x * SEED1 + p1.y / (SEED1 + 0.15555));
}

fn ditherBlueNoise3(color: vec3f, xy: vec2f, d: vec3f) -> vec3f {
    let decimated = decimate3(color, d);
    let diff = (color - decimated) * d;
    return decimate3(color + step(vec3(ditherBlueNoise2(xy)), diff) / d, d);
}