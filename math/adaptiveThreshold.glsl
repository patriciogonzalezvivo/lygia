/*
original_author: [Brad Larson, Ben Cochran, Hugues Lismonde, Keitaroh Kobayashi, Alaric Cole, Matthew Clark, Jacob Gundersen, Chris Williams]
description: adaptive threshold from https://github.com/BradLarson/GPUImage/blob/master/framework/Source/GPUImageAdaptiveThresholdFilter.m
use: adaptiveThreshold(<float> value, <float> blur_value[,<float> bias])
*/

#ifndef FNC_ADAPTIVETHRESHOLD
#define FNC_ADAPTIVETHRESHOLD
float adaptiveThreshold(in float value, in float blur_value, in float bias) {
    return step(blur_value + bias, value);
}

float adaptiveThreshold(in float value, in float blur_value) {
    return step(blur_value - 0.05, value);
}
#endif
