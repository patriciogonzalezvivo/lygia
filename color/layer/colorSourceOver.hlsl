#include "../blend.hlsl"
#include "../composite/sourceOver.hlsl"

/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Color Blending with Porter Duff Source Over Compositing
use: <float4> layerColorSourceOver(<float4> src, <float4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LAYER_COLOR_SRC_OVER
#define FNC_LAYER_COLOR_SRC_OVER

float4 layerColorSourceOver(float4 src, float4 dest) {
    float4 result = float4(0.0, 0.0, 0.0, 0.0);

    // Compute color for RGB channels
    float3 blendedColor = blendColor(src.rgb, dest.rgb);

    // Compute source-over for RGB channels
    result.rgb =  compositeSourceOver(blendedColor, dest.rgb, src.a, dest.a);

    // Compute source-over for the alpha channel
    result.a = compositeSourceOver(src.a, dest.a);

    return result;
}
#endif
