#include "../sample/sprite.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample a frame on a sprite sheet
use: <SPRITELOOP_TYPE> SpriteLOOP(<texture_2d<f32>>tex, <sampler> samp, <vec2f> st, <vec2f> grid, vec2f, <f32> start_index, <f32> end_index, <f32> time)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn spriteLoop(tex: texture_2d<f32>, samp: sampler, st: vec2f, grid: vec2f, start_index: f32, end_index: f32, time: f32) -> vec4f {
  let frame = time % (end_index - start_index);
  return sampleSprite(tex, samp, st, grid, start_index + frame);
}
