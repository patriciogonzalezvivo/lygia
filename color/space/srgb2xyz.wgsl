#include "rgb2xyz.wgsl"
#include "srgb2rgb.wgsl"

fn srgb2xyz(srgb: vec3f) -> vec3f { return rgb2xyz(srgb2rgb(srgb)); }
