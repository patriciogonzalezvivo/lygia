#include "rgb2lab.glsl"
#include "srgb2rgb.glsl"

fn srgb2lab(srgb: vec3f) -> vec3f { return rgb2lab(srgb2rgb(srgb));}

