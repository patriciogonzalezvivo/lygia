/*
contributors: [Brad Larson, Ben Cochran, Hugues Lismonde, Keitaroh Kobayashi, Alaric Cole, Matthew Clark, Jacob Gundersen, Chris Williams]
description: adaptive threshold from https://github.com/BradLarson/GPUImage/blob/master/framework/Source/GPUImageAdaptiveThresholdFilter.m
*/

fn adaptiveThreshold(v: f32, blur_v: f32, b: f32) -> f32 { return step(blur_v + b, v); }
