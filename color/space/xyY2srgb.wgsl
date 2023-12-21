#include "xyz2srgb.wgsl"
#include "xyY2xyz.wgsl"

fn xyY2srgb(xyY: vec3f) -> vec3f { return xyz2srgb(xyY2xyz(xyY)); }