#include "srgb2rgb.wgsl"
#include "rgb2oklab.wgsl"

fn srgb2oklab(srgb: vec3f) -> vec3f { return rgb2oklab( srgb2rgb(srgb) ); }
