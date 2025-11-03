/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination In Compositing
use: compositeDestinationIn(<vec4f> src, <vec4f> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn compositeDestinationIn(src: f32, dst: f32) -> f32 {
    return dst * src;
}

fn compositeDestinationIn3(srcColor: vec3f, dstColor: vec3f, srcAlpha: f32, dstAlpha: f32) -> vec3f {
    return dstColor * srcAlpha;
}

fn compositeDestinationIn4(srcColor: vec4f, dstColor: vec4f) -> vec4f {
    return vec4f(
        compositeDestinationIn3(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a),
        compositeDestinationIn(srcColor.a, dstColor.a)
    );
}
