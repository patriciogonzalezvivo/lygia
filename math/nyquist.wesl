/*
contributors: Shadi El Hajj
description: A simple low-pass filter which attenuates high-frequencies.
use: nyquist(value: f32, width: f32) -> f32
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

const NYQUIST_FILTER_CENTER: f32 = 0.5;
const NYQUIST_FILTER_WIDTH: f32 = 0.25;

fn nyquist(x: f32, width: f32) -> f32 {
    let cutoffStart = NYQUIST_FILTER_CENTER - NYQUIST_FILTER_WIDTH;
    let cutoffEnd = NYQUIST_FILTER_CENTER + NYQUIST_FILTER_WIDTH;
    let f = smoothstep(cutoffEnd, cutoffStart, width);
    return mix(0.5, x, f);
}
