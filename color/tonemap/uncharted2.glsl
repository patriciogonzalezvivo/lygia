/*
author: John Hable
description: Tonemapping function from presentation, uncharted 2 HDR Lighting, Page 142 to 143
use: <vec3|vec4> tonemapUncharted2(<vec3|vec4> x)
*/

#ifndef FNC_TONEMAPUNCHARTED2
#define FNC_TONEMAPUNCHARTED2
vec3 tonemapUncharted2(vec3 v) {
    float A = 0.15; // 0.22
    float B = 0.50; // 0.30
    float C = 0.10;
    float D = 0.20;
    float E = 0.02; // 0.01
    float F = 0.30;
    float W = 11.2;
    
    vec4 x = vec4(v, W);
    x = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
    return x.xyz / x.w;
}

vec4 tonemapUncharted2(const vec4 x) { return vec4( tonemapUncharted2(x.rgb), x.a); }
#endif