#include "lch2lab.wgsl"
#include "lab2rgb.wgsl"

fn lch2rgb(lch: vec3f) -> vec3f { return lab2rgb( lch2lab(lch) ); }
