/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination Over Compositing
use: <vec4> compositeDestinationOver(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn compositeDestinationOver(src: f32, dst: f32) -> f32 {
    return dst + src * (1.0 - dst);
}

fn compositeDestinationOver3(srcColor: vec3f, dstColor: vec3f, srcAlpha: f32, dstAlpha: f32) -> vec3f {
    return dstColor + srcColor * (1.0 - dstAlpha);
}

fn compositeDestinationOver4(srcColor: vec4f, dstColor: vec4f) -> vec4f {
    let result = vec4f(0.0);

    result.rgb = compositeDestinationOver(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeDestinationOver(srcColor.a, dstColor.a);

    return result;
}
