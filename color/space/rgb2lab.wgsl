#include "rgb2xyz.wgsl"
#include "xyz2lab.wgsl"

fn rgb2lab(rgb: vec3f ) -> vec3f { return xyz2lab(rgb2xyz(rgb)); }
