#include "oklab2srgb.wgsl"
#include "srgb2rgb.wgsl"

/*
contributors: Bjorn Ottosson (@bjornornorn)
description: oklab to linear RGB https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> oklab2rgb(<vec3|vec4> oklab)
*/

fn oklab2rgb(oklab: vec3f) -> vec3f { return srgb2rgb(oklab2srgb(oklab)); }
