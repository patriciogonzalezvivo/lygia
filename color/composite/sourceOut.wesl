/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source Out Compositing
use: compositeSourceOut(<vec4f> src, <vec4f> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn compositeSourceOut(src: f32, dst: f32) -> f32 {
    return src * (1.0 - dst);
}

fn compositeSourceOut3(srcColor: vec3f, dstColor: vec3f, srcAlpha: f32, dstAlpha: f32) -> vec3f {
    return srcColor * (1.0 - dstAlpha);
}

fn compositeSourceOut4(srcColor: vec4f, dstColor: vec4f) -> vec4f {
    return vec4f(
        compositeSourceOut3(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a),
        compositeSourceOut(srcColor.a, dstColor.a)
    );
}
