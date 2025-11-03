import lygia::color::blend::average::blendAverage3;
import lygia::color::composite::sourceOver::{compositeSourceOver, compositeSourceOver3};

/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Average Blending with Porter Duff Source Over Compositing
use: layerAverageSourceOver4(<vec4f> src, <vec4f> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn layerAverageSourceOver4(src: vec4f, dst: vec4f) -> vec4f {
    
    // Compute average for RGB channels
    let blendedColor = blendAverage3(src.rgb, dst.rgb);

    // Compute source-over for RGB channels
    let rgb_result = compositeSourceOver3(blendedColor, dst.rgb, src.a, dst.a);

    // Compute source-over for the alpha channel
    let a_result = compositeSourceOver(src.a, dst.a);

    return vec4f(rgb_result, a_result);
}
