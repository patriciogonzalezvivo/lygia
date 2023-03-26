/*
original_author: Patricio Gonzalez Vivo
description: |
    Physical Hue. Ratio: 1/3 = neon, 1/4 = refracted, 1/5+ = approximate white
use: <float3> hue(<float> hue[, <float> ratio])
*/

#ifndef FNC_PALETTE_HUE
#define FNC_PALETTE_HUE

float3 hue(float _hue, float _ratio) {
    return smoothstep(  
                        float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0), 
                        // float3(0.9059, 0.8745, 0.8745), float3(1.0, 1.0, 1.0),
                        abs( ((_hue + float3(0.0, 1.0, 2.0) * _ratio) % 1.0) * 2.0 - 1.0));
}

float3 hue(float _hue) { return hue(_hue, 0.33333); }

#endif