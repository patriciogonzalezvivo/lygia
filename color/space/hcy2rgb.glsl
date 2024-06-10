#include "hue2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a HCY color to linear RGB'
use: <vec3|vec4> hcy2rgb(<vec3|vec4> hsl)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HCY2RGB
#define FNC_HCY2RGB
vec3 hcy2rgb(vec3 hcy) {
    const vec3 HCYwts = vec3(0.299, 0.587, 0.114);
    vec3 RGB = hue2rgb(hcy.x);
    float Z = dot(RGB, HCYwts);
    if (hcy.z < Z) {
        hcy.y *= hcy.z / Z;
    } else if (Z < 1.0) {
        hcy.y *= (1.0 - hcy.z) / (1.0 - Z);
    }
    return (RGB - Z) * hcy.y + hcy.z;
}
vec4 hcy2rgb(vec4 hcy) { return vec4(hcy2rgb(hcy.rgb), hcy.a); }
#endif