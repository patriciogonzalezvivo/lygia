import lygia::math::cubicMix::cubicMix3;
import lygia::math::{mmin::mmin3, mmax::mmax3};

/*
contributors: Patricio Gonzalez Vivo
description: |
    Convert from RYB to RGB color space.
    Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
    and http://vis.computer.org/vis2004/DVD/infovis/papers/gossett.pdf
use: <vec3|vec4> ryb2rgb(<vec3|vec4> ryb)
note: |
    Two implementations are provided via WESL conditionals:
    - @if(RYB_FAST): Non-homogeneous version (faster)
    - @if(!RYB_FAST): Homogeneous version using cubic interpolation (default)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(RYB_FAST)
fn ryb2rgb(ryb_in: vec3f) -> vec3f {
    var ryb = ryb_in;
    // Remove the white from the color
    let w = mmin3(ryb);
    ryb -= w;

    let max_y = mmax3(ryb);

    // Get the green out of the yellow & blue
    let g = min(ryb.g, ryb.b);
    var rgb = ryb - vec3f(0.0, g, g);

    if (rgb.b > 0.0 && g > 0.0) {
        rgb.b *= 2.0;
        var g_mut = g * 2.0;
    }

    // Redistribute the remaining yellow.
    rgb.r += rgb.g;
    rgb.g += g;

    // Normalize to values.
    let max_g = mmax3(rgb);
    rgb *= select(1.0, max_y / max_g, max_g > 0.0);

    // Add the white back in.
    return rgb + w;
}

@if(!RYB_FAST)
fn ryb2rgb(ryb: vec3f) -> vec3f {
    let ryb000 = vec3f(1., 1., 1.);       // White
    let ryb001 = vec3f(.163, .373, .6);   // Blue
    let ryb010 = vec3f(1., 1., 0.);       // Yellow
    let ryb100 = vec3f(1., 0., 0.);       // Red
    let ryb011 = vec3f(0., .66, .2);      // Green
    let ryb101 = vec3f(.5, 0., .5);       // Violet
    let ryb110 = vec3f(1., .5, 0.);       // Orange
    let ryb111 = vec3f(0., 0., 0.);       // Black
    return
        cubicMix3(
            cubicMix3(
                cubicMix3(ryb000, ryb001, vec3f(ryb.z)),
                cubicMix3(ryb010, ryb011, vec3f(ryb.z)),
                vec3f(ryb.y)
            ),
            cubicMix3(
                cubicMix3(ryb100, ryb101, vec3f(ryb.z)),
                cubicMix3(ryb110, ryb111, vec3f(ryb.z)),
                vec3f(ryb.y)
            ),
            vec3f(ryb.x)
        );
}

fn ryb2rgb4(ryb: vec4f) -> vec4f {
    return vec4f(ryb2rgb(ryb.rgb), ryb.a);
}
