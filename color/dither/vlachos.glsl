/*
original_author: Patricio Gonzalez Vivo
description: Vlachos 2016, "Advanced VR Rendering"
use: <vec4|vec3|float> ditherVlachos(<vec4|vec3|float> value, <float> time)
*/

#ifndef DITHER_VLACHOS
#define DITHER_VLACHOS

float ditherVlachos(float b, float time) {
    HIGHP float noise = dot(vec2(171.0, 231.0), gl_FragCoord.xy + time);
    noise = fract(noise / 71.0);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return b + (noise / 255.0);
}

vec3 ditherVlachos(vec3 rgb, float time) {
    HIGHP vec3 noise = vec3(dot(vec2(171.0, 231.0), gl_FragCoord.xy + time));
    noise = fract(noise / vec3(103.0, 71.0, 97.0));
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return rgb.rgb + (noise / 255.0);
}

vec4 ditherVlachos(vec4 rgba, float time) {
    return vec4(ditherVlachos(rgba.rgb, time), rgba.a);
}

#endif