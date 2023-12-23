/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a hue value to a RGB float3 color.
use: <float3> hue2rgb(<float> hue)
*/

#ifndef FNC_HUE2RGB
#define FNC_HUE2RGB
float3 hue2rgb(float hue) {
    float R = abs(hue * 6.0 - 3.0) - 1.0;
    float G = 2.0 - abs(hue * 6.0 - 2.0);
    float B = 2.0 - abs(hue * 6.0 - 4.0);
    return saturate(float3(R,G,B));
}
#endif