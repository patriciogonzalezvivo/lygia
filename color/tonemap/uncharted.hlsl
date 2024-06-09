/*
contributors: nan
description: 'Uncharted 2 Tonemapping'
use: <float3|float4> tonemapUncharted(<float3|float4> x)
*/

#ifndef FNC_TONEMAPUNCHARTED
#define FNC_TONEMAPUNCHARTED

float3 uncharted2Tonemap(const float3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

float3 tonemapUncharted(const float3 x) {
    const float W = 11.2;
    const float exposureBias = 2.0;
    float3 curr = uncharted2Tonemap(exposureBias * x);
    float3 whiteScale = 1.0 / uncharted2Tonemap(float3(W, W, W));
    return curr * whiteScale;
}

float4 tonemapUncharted(const float4 x) { return float4( tonemapUncharted(x.rgb), x.a); }
#endif