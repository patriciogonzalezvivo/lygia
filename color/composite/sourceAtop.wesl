/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source Atop Compositing
use: compositeSourceAtop(<vec4f> src, <vec4f> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn compositeSourceAtop(src: f32, dst: f32) -> f32 {
    return src * dst + dst * (1.0 - src);
}

fn compositeSourceAtop3(srcColor: vec3f, dstColor: vec3f, srcAlpha: f32, dstAlpha: f32) -> vec3f {
    return srcColor * dstAlpha + dstColor * (1.0 - srcAlpha);
}

fn compositeSourceAtop4(srcColor: vec4f, dstColor: vec4f) -> vec4f {
    return vec4f(
        compositeSourceAtop3(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a),
        compositeSourceAtop(srcColor.a, dstColor.a)
    );
}
