/*
contributors: [Brad Larson, Ben Cochran, Hugues Lismonde, Keitaroh Kobayashi, Alaric Cole, Matthew Clark, Jacob Gundersen, Chris Williams]
description: adaptive threshold from https://github.com/BradLarson/GPUImage/blob/master/framework/Source/GPUImageAdaptiveThresholdFilter.m
use: adaptiveThreshold(<float> value, <float> blur_value [,<float> bias])
*/

#ifndef FNC_ADAPTIVETHRESHOLD
#define FNC_ADAPTIVETHRESHOLD
float adaptiveThreshold(in float v, in float blur_v, in float b) {
    return step(blur_v + b, v);
}

float adaptiveThreshold(in float v, in float blur_v) {
    return step(blur_v - 0.05, v);
}
#endif
