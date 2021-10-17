/*
Author: Unreal Engine 4.0
description:  Adapted to be close to TonemapACES, with similar range. Gamma 2.2 correction is baked in, don't use with sRGB conversion! https://docs.unrealengine.com/4.26/en-US/RenderingAndGraphics/PostProcessEffects/ColorGrading/
use: tonemapUnreal(<vec3> x)
*/

#ifndef FNC_TONEMAPUNREAL
#define FNC_TONEMAPUNREAL
vec3 tonemapUnreal(const vec3 x) { return x / (x + 0.155) * 1.019; }
#endif