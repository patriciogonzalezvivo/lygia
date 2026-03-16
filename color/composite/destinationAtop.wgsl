/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination Atop Compositing
use: <vec4> compositeDestinationAtop(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn compositeDestinationAtop(src: f32, dst: f32) -> f32 {
    return dst * src + src * (1.0 - dst);
}

fn compositeDestinationAtop3(srcColor: vec3f, dstColor: vec3f, srcAlpha: f32, dstAlpha: f32) -> vec3f {
    return dstColor * srcAlpha + srcColor * (1.0 - dstAlpha);
}

fn compositeDestinationAtop4(srcColor: vec4f, dstColor: vec4f) -> vec4f {
    let result = vec4f(0.0);
   
    result.rgb = compositeDestinationAtop(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeDestinationAtop(srcColor.a, dstColor.a);

    return result;
}
