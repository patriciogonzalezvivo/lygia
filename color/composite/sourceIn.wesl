/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source In Compositing
use: compositeSourceIn(<vec4f> src, <vec4f> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn compositeSourceIn(src: f32, dst: f32) -> f32 {
    return src * dst;
}

fn compositeSourceIn3(srcColor: vec3f, dstColor: vec3f, srcAlpha: f32, dstAlpha: f32) -> vec3f {
    return srcColor * dstAlpha;
}

fn compositeSourceIn4(srcColor: vec4f, dstColor: vec4f) -> vec4f {
    return vec4f(
        compositeSourceIn3(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a),
        compositeSourceIn(srcColor.a, dstColor.a)
    );
}
