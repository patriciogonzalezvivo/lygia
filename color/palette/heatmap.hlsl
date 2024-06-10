/*
contributors: Patricio Gonzalez Vivo
description: Heatmap palette
use: <float3> heatmap(<float> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HEATMAP
#define FNC_HEATMAP
float3 heatmap(float v) {
    float3 r = v * 2.1 - float3(1.8, 1.14, 0.3);
    return 1.0 - r * r;
}
#endif