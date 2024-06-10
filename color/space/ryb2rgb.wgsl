#include "../../math/mmin.wgsl"
#include "../../math/mmax.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from RYB to RGB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <vec3|vec4> ryb2rgb(<vec3|vec4> ryb)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn ryb2rgb(ryb: vec3f) -> vec3f {
    // Remove white component
    let v = ryb - mmin(ryb);
    
    // Derive rgb
    let yb = min(v.y, v.b);
    var rgb = vec3f(0.0, 0.0, 0.0);
    rgb.r = v.r + v.y - yb;
    rgb.g = v.y + (2.0 * yb);
    rgb.b = 2.0 * (v.b - yb);
    
    // Normalize
    let n = mmax(rgb) / mmax(v);
    if (n > 0.0)
        rgb /= n;
    
    // Add black
    return rgb + mmin(1.0 - ryb);
}
