#include "xyz2srgb.wgsl"
#include "srgb2rgb.wgsl"

fn xyz2rgb(xyz: vec3f) -> vec3f { return srgb2rgb(xyz2srgb(xyz)); }