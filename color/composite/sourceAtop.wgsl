/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source Atop Compositing
use: <vec4> compositeSourceAtop(<vec4> src, <vec4> dst)
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
    let result = vec4f(0.0);

    result.rgb = compositeSourceAtop(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeSourceAtop(srcColor.a, dstColor.a);

    return result;
}
