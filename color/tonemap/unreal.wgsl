/*
contributors: Unreal Engine 4.0
description:  Adapted to be close to TonemapACES, with similar range. Gamma 2.2 correction is baked in, don't use with sRGB conversion! https://docs.unrealengine.com/4.26/en-US/RenderingAndGraphics/PostProcessEffects/ColorGrading/
use: <vec3|vec4> tonemapUnreal(<vec3|vec4> x)
*/

fn tonemapUnreal3(x: vec3f) -> vec3f { return x / (x + 0.155) * 1.019; }
fn tonemapUnreal4(x: vec4f) -> vec4f { return vec4f(tonemapUnreal(x.rgb), x.a); }
