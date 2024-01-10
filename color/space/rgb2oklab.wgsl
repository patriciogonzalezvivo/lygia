#include "rgb2srgb.wgsl"
#include "srgb2oklab.wgsl"

/*
contributors: Bjorn Ottosson (@bjornornorn)
description: |
    Linear rgb ot OKLab https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> rgb2oklab(<vec3|vec4> srgb)
*/

fn rgb2oklab(rgb: vec3f) -> vec3f { srgb2oklab( rgb2srgb(rgb) ); }