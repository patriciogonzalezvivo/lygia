#include "../space/sprite.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample a frame on a sprite sheet
use: <vec4f> sampleSprite(<texture_2d<f32>>tex, <sampler> samp, <vec2f> st, <vec2f> grid, <f32> frame)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleSprite(tex: texture_2d<f32>, samp: sampler, st: vec2f, grid: vec2f, frame: f32) -> vec4f {
  return textureSample(tex, samp, sprite(st, grid, frame));
}
