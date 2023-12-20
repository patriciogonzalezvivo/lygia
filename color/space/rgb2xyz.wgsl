#include "srgb2xyz.wgsl"
#include "rgb2srgb.wgsl"
fn rgb2xyz(rgb: vec3f) -> vec3f { return SRGB2XYZ * rgb2srgb(rgb);}