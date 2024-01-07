#include "../sample.glsl"
#include "../math/const.glsl"
#include "../color/space/rgb2hsv.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample an optical flow direction where the angle is encoded in the Hue and magnitude in the Saturation
use: sampleFlow(<SAMPLER_TYPE> tex, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/


#ifndef SAMPLEOPTICALFLOW_SAMPLE_FNC
#define SAMPLEOPTICALFLOW_SAMPLE_FNC(TEX, UV) texture2D(TEX, UV).rgb
#endif

#ifndef FNC_SAMPLEOPTICALFLOW
#define FNC_SAMPLEOPTICALFLOW
vec2 sampleOpticalFlow(sampler2D tex, vec2 st) {
    vec3 data = SAMPLEOPTICALFLOW_SAMPLE_FNC(tex, st);
    vec3 hsv = rgb2hsv(data.rgb);
    float a = (hsv.x * 2.0 - 1.0) * PI;
    return vec2(cos(a), -sin(a)) * hsv.y;
}

vec2 sampleOpticalFlow(sampler2D tex, vec2 st, vec2 resolution, float max_dist) { return sampleOpticalFlow(tex, st) * (max_dist / resolution); }
#endif