/*
contributors: nan
description: Uncharted 2 tonemapping operator
use: <vec3|vec4> tonemapUncharted(<vec3|vec4> x)
*/

#ifndef FNC_TONEMAPUNCHARTED
#define FNC_TONEMAPUNCHARTED

vec3 uncharted2Tonemap(const vec3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemapUncharted(const vec3 x) {
    const float W = 11.2;
    const float exposureBias = 2.0;
    vec3 curr = uncharted2Tonemap(exposureBias * x);
    vec3 whiteScale = 1.0 / uncharted2Tonemap(vec3(W));
    return curr * whiteScale;
}

vec4 tonemapUncharted(const vec4 x) { return vec4( tonemapUncharted(x.rgb), x.a); }
#endif