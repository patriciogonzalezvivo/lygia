/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Xor Compositing
use: compositeXor(<vec4f> src, <vec4f> dst)
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
    return vec4f(
        compositeXor3(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a),
        compositeXor(srcColor.a, dstColor.a)
    );
}
