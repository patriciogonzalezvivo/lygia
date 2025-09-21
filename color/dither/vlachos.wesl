import package::math::decimate::decimate3;

/*
contributors: Patricio Gonzalez Vivo
description: 'Vlachos 2016, "Advanced VR Rendering" http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf'
use: <vec4|vec3|float> ditherVlachos(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_VLACHOS_TIME
    - DITHER_VLACHOS_CHROMATIC
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn ditherVlachos1(ist: vec2f) -> vec3f {
    var st = ist;
    st += 1337.0 * fract(uniforms.frameIdx);
    var noise = vec3(dot(vec2(171.0, 231.0), st));
    noise = fract(noise / vec3(103.0, 71.0, 97.0));
    return noise;
}

fn ditherVlachos3(color: vec3f, st: vec2f, d: vec3f) -> vec3f {
    let ditherPattern = ditherVlachos1(st);
    return decimate3(color + ditherPattern / d, d);
}