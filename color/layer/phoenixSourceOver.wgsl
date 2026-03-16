#include "../blend.wgsl"
#include "../composite/sourceOver.wgsl"

/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Phoenix Blending with Porter Duff Source Over Compositing
use: <vec4> layerPhoenixSourceOver(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn layerPhoenixSourceOver(src: vec4f, dest: vec4f) -> vec4f {
    let result = vec4f(0.0, 0.0, 0.0, 0.0);

    // Compute phoenix for RGB channels
    let blendedColor = blendPhoenix(src.rgb, dest.rgb);

    // Compute source-over for RGB channels
    result.rgb = compositeSourceOver(blendedColor, dest.rgb, src.a, dest.a);

    // Compute source-over for the alpha channel
    result.a = compositeSourceOver(src.a, dest.a);

    return result;
}
