#include "rgb2lch.wgsl"
#include "srgb2rgb.wgsl"

fn srgb2lch(srgb: vec3f) -> vec3f { return rgb2lch(srgb2rgb(srgb)); }
