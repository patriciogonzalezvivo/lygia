/*
contributors: Unreal Engine 4.0
description: Adapted to be close to TonemapACES, with similar range. Gamma 2.2 correction is baked in, don't use with sRGB conversion! https://docs.unrealengine.com/4.26/en-US/RenderingAndGraphics/PostProcessEffects/ColorGrading/
use: <float3|float4> tonemapUnreal(<float3|float4> x)
*/

#ifndef FNC_TONEMAPUNREAL
#define FNC_TONEMAPUNREAL
float3 tonemapUnreal(const float3 x) { return x / (x + 0.155) * 1.019; }
float4 tonemapUnreal(const float4 x) { return float4(tonemapUnreal(x.rgb), x.a); }
#endif