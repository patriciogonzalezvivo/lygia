/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Xor Compositing
use: <vec4> compositeXor(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn compositeXor(src: f32, dst: f32) -> f32 {
    return src * (1.0 - dst) + dst * (1.0 - src);
}

fn compositeXor3(srcColor: vec3f, dstColor: vec3f, srcAlpha: f32, dstAlpha: f32) -> vec3f {
    return srcColor * (1.0 - dstAlpha) + dstColor * (1.0 - srcAlpha);
}

fn compositeXor4(srcColor: vec4f, dstColor: vec4f) -> vec4f {
    let result = vec4f(0.0);
    result.rgb = compositeXor(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeXor(srcColor.a, dstColor.a);
    return result;
}
