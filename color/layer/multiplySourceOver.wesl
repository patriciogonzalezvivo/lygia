import lygia::color::blend::multiply::blendMultiply3;
import lygia::color::composite::sourceOver::{compositeSourceOver, compositeSourceOver3};

/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Multiply Blending with Porter Duff Source Over Compositing
use: layerMultiplySourceOver4(<vec4f> src, <vec4f> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn layerMultiplySourceOver4(src: vec4f, dst: vec4f) -> vec4f {
    // Compute multiply for RGB channels
    let blendedColor = blendMultiply3(src.rgb, dst.rgb);

    // Compute source-over for RGB and alpha channels
    return vec4f(
        compositeSourceOver3(blendedColor, dst.rgb, src.a, dst.a),
        compositeSourceOver(src.a, dst.a)
    );
}
