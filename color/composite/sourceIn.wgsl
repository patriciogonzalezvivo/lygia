/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source In Compositing
use: <vec4> compositeSourceIn(<vec4> src, <vec4> dst)
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
    let result = vec4f(0.0);

    result.rgb = compositeSourceIn(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeSourceIn(srcColor.a, dstColor.a);

    return result;
}
