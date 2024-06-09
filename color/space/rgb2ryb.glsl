#include "../../math/mmin.glsl"
#include "../../math/mmax.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a color from RGB to RYB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <vec3> ryb2rgb(<vec3> ryb)
examples:
  - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
  - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
  - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2RYB
#define FNC_RGB2RYB

vec3 rgb2ryb(vec3 rgb) {
    // Remove white component
    vec3 v = rgb - mmin(rgb);

    // Derive ryb
    vec3 ryb = vec3(0.0, 0.0, 0.0);
    float rg = min(v.r, v.g);
    ryb.r = v.r - rg;
    ryb.y = 0.5 * (v.g + rg);
    ryb.b = 0.5 * (v.b + v.g - rg);
    
    // Normalize
    float n = mmax(ryb) / mmax(v);
    if (n > 0.0)
    	ryb /= n;
    
    // Add black 
    return ryb + mmin(1.0 - rgb);
}

vec4 rgb2ryb(vec4 rgb) { return vec4(rgb2ryb(rgb.rgb), rgb.a); }

#endif
