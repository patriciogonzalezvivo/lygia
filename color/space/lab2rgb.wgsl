#include "lab2xyz.wgsl"
#include "xyz2rgb.wgsl"

fn lab2rgb(lab : vec3f) -> vec3f { return xyz2rgb( lab2xyz( lab ) ); }
