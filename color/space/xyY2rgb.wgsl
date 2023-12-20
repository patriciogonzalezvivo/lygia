#include "xyz2rgb.wgsl"
#include "xyY2xyz.wgsl"

fn xyY2rgb(xyY: vec3f) -> vec3f { return xyz2rgb(xyY2xyz(xyY)); }