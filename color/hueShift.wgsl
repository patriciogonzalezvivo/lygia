#include "space/hsl2rgb.glsl"
#include "space/rgb2hsl.glsl"

/*
contributors:
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Shifts color hue
use: <vec3f> hueShift(<vec3f> color, <float> angle)
optionas:
    - HUESHIFT_AMOUNT: if defined, it uses a normalized value instead of an angle
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hueShift(color: vec3f, a: f32 ) -> vec3f{
    var hsl = rgb2hsl(color);
    hsl.r = hsl.r * TAU + a;
    hsl.r = fract(hsl.r / TAU);
    return hsl2rgb(hsl);
}