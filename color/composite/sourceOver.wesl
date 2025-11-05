/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source Over Compositing
use: compositeSourceOver(<vec4f> src, <vec4f> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn compositeSourceOver(src: f32, dst: f32) -> f32 {
    return src + dst * (1.0 - src);
}

fn compositeSourceOver3(srcColor: vec3f, dstColor: vec3f, srcAlpha: f32, dstAlpha: f32) -> vec3f {
    return srcColor * srcAlpha + dstColor * dstAlpha * (1.0 - srcAlpha);
}

fn compositeSourceOver4(srcColor: vec4f, dstColor: vec4f) -> vec4f {
    return vec4f(
        compositeSourceOver3(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a),
        compositeSourceOver(srcColor.a, dstColor.a)
    );
}
