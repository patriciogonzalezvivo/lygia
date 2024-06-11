#include "../../math/mmin.wgsl"
#include "../../math/mmax.wgsl"
#include "../../math/cubicMix.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a color from RGB to RYB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <vec3f> ryb2rgb(<vec3f> ryb)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2ryb(rgb: vec3f) -> vec3f {
    let rgb000 = vec3f(1., 1., 1.);       // Black
    let rgb100 = vec3f(1., 0., 0.);       // Red          
    let rgb010 = vec3f(0., 1., .483);     // Green
    let rgb110 = vec3f(0., 1., 0.);       // Yellow
    let rgb001 = vec3f(0., 0., 1.);       // Blue
    let rgb101 = vec3f(.309, 0., .469);   // Magenta
    let rgb011 = vec3f(0., .053, .210);   // Turquoise
    let rgb111 = vec3f(0., 0., 0.);       // White
    return cubeMix3(cubeMix3(
        cubeMix3(rgb000, rgb001, rgb.z),
        cubeMix3(rgb010, rgb011, rgb.z),
        rgb.y), cubeMix3(
        cubeMix3(rgb100, rgb101, rgb.z),
        cubeMix3(rgb110, rgb111, rgb.z),
        rgb.y), rgb.x);
}