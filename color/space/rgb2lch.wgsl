#include "rgb2lab.wgsl"
#include "lab2lch.wgsl"

fn rgb2lch(rgb: vec3f ) -> vec3f { return lab2lch(rgb2lab(rgb)); }