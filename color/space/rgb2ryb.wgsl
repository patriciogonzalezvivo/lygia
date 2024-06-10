#include "../../math/mmin.wgsl"
#include "../../math/mmax.wgsl"

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
    // Remove white component
    let v = rgb - mmin(rgb);

    // Derive ryb
    var ryb = vec3f(0.0, 0.0, 0.0);
    float rg = min(v.r, v.g);
    ryb.r = v.r - rg;
    ryb.y = 0.5 * (v.g + rg);
    ryb.b = 0.5 * (v.b + v.g - rg);
    
    // Normalize
    let n = mmax(ryb) / mmax(v);
    if (n > 0.0)
    	ryb /= n;
    
    // Add black 
    return ryb + mmin(1.0 - rgb);
}
