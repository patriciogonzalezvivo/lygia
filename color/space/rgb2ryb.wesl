import lygia::math::cubicMix::cubicMix3;
import lygia::math::{mmin::mmin3, mmax::mmax3};

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a color from RGB to RYB color space.
    Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
    and https://bahamas10.github.io/ryb/assets/ryb.pdf
use: <vec3|vec4> rgb2ryb(<vec3|vec4> rgb)
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
fn rgb2ryb(rgb_in: vec3f) -> vec3f {
    var rgb = rgb_in;
    // Remove the white from the color
    let w = mmin3(rgb);
    let bl = mmin3(vec3f(1.0) - rgb);
    rgb -= w;

    let max_g = mmax3(rgb);

    // Get the yellow out of the red & green
    let y = min(rgb.r, rgb.g);
    var ryb = rgb - vec3f(y, y, 0.0);

    // If this unfortunate conversion combines blue and green, then cut each in half to preserve the value's maximum range.
    if (ryb.b > 0.0 && ryb.y > 0.0) {
        ryb.b *= 0.5;
        ryb.y *= 0.5;
    }

    // Redistribute the remaining green.
    ryb.b += ryb.y;
    ryb.y += y;

    // Normalize to values.
    let max_y = mmax3(ryb);
    ryb *= select(1.0, max_g / max_y, max_y > 0.0);

    // Add the white back in.
    return ryb + bl;
}

@if(!RYB_FAST)
fn rgb2ryb(rgb: vec3f) -> vec3f {
    let rgb000 = vec3f(1., 1., 1.);       // Black
    let rgb100 = vec3f(1., 0., 0.);       // Red
    let rgb010 = vec3f(0., 1., .483);     // Green
    let rgb110 = vec3f(0., 1., 0.);       // Yellow
    let rgb001 = vec3f(0., 0., 1.);       // Blue
    let rgb101 = vec3f(.309, 0., .469);   // Magenta
    let rgb011 = vec3f(0., .053, .210);   // Turquoise
    let rgb111 = vec3f(0., 0., 0.);       // White
    return cubicMix3(
        cubicMix3(
            cubicMix3(rgb000, rgb001, vec3f(rgb.z)),
            cubicMix3(rgb010, rgb011, vec3f(rgb.z)),
            vec3f(rgb.y)),
        cubicMix3(
            cubicMix3(rgb100, rgb101, vec3f(rgb.z)),
            cubicMix3(rgb110, rgb111, vec3f(rgb.z)),
            vec3f(rgb.y)),
        vec3f(rgb.x));
}

fn rgb2ryb4(rgb: vec4f) -> vec4f {
    return vec4f(rgb2ryb(rgb.rgb), rgb.a);
}
