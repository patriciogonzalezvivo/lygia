#include "xyz2rgb.wgsl"
#include "rgb2srgb.wgsl"

fn xyz2srgb(xyz: vec3f) -> vec3f { return rgb2srgb(xyz2rgb(xyz)); }