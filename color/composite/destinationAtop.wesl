/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination Atop Compositing
use: compositeDestinationAtop(<vec4f> src, <vec4f> dst)
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
    return vec4f(
        compositeDestinationAtop3(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a),
        compositeDestinationAtop(srcColor.a, dstColor.a)
    );
}
