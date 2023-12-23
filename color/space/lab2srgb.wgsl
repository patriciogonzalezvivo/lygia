#include "lab2xyz.wgsl"
#include "xyz2srgb.wgsl"

fn lab2srgb(lab: vec3f) -> vec3f { return xyz2srgb( lab2xyz( lab ) ); }
