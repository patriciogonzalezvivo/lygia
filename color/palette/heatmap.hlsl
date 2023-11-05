/*
contributors: Patricio Gonzalez Vivo
description: heatmap palette
use: <float3> heatmap(<float> value)
*/

#ifndef FNC_HEATMAP
#define FNC_HEATMAP
float3 heatmap(float v) {
    float3 r = v * 2.1 - float3(1.8, 1.14, 0.3);
    return 1.0 - r * r;
}
#endif