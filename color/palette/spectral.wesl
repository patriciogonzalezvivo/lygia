/*
contributors: Martijn Steinrucken
description: Spectral Response Function https://www.shadertoy.com/view/wlSBzD
use: <vec3f> spectral(<f32> value)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
license: MIT License Copyright (c) 2020 Martijn Steinrucken
*/

import lygia::math::consts::PI;

fn spectral(x: f32) -> vec3f {
    return (vec3f( 1.220023e0,-1.933277e0, 1.623776e0) +
           (vec3f(-2.965000e1, 6.806567e1,-3.606269e1) +
           (vec3f( 5.451365e2,-7.921759e2, 6.966892e2) +
           (vec3f(-4.121053e3, 4.432167e3,-4.463157e3) +
           (vec3f( 1.501655e4,-1.264621e4, 1.375260e4) +
           (vec3f(-2.904744e4, 1.969591e4,-2.330431e4) +
           (vec3f( 3.068214e4,-1.698411e4, 2.229810e4) +
           (vec3f(-1.675434e4, 7.594470e3,-1.131826e4) +
            vec3f( 3.707437e3,-1.366175e3, 2.372779e3)
           *x)*x)*x)*x)*x)*x)*x)*x)*x;
}

fn spectralExpand(x_in: f32, l: f32) -> vec3f {
    var x = 1.0 - x_in;
    // (optional) rectangular expand
    x = mix((x * 0.5) + 0.25, x, 1.0 - l);
    return vec3f(
        // RED + VIOLET-FALLOFF
        -0.0833333 * (l - 1.0) * (
            cos(PI * max(0.0, min(1.0, 12.0 * abs((0.0833333 * l + x - 0.8333333) / (l + 2.0))))) + 1.0
        ) + 0.5 * cos(PI * min(1.0, (l + 3.0) * abs(-0.1666666 * l + x - 0.3333333))) + 0.5,
        // GREEN, BLUE
        0.5 + 0.5 * cos(PI * min(
            vec2f(1.0),
            abs(vec2f(x) - vec2f(0.5, (1.0 - ((2.0 + l) / 3.0) * 0.5))) * vec2f(3.0 + l)
        ))
    );
}
