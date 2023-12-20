#include "rgb2xyz.wgsl"
#include "xyz2xyY.wgsl"

fn rgb2xyY(rgb: vec3f) -> vec3f { return xyz2xyY(rgb2xyz(rgb)); }