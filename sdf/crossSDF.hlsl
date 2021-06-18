#include "rectSDF.hlsl"

/*
author: Patricio Gonzalez Vivo
description: Returns a cross-shaped SDF
use: crossSDF(<float2> st, size s)
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
    Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
    https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_CROSSSDF
#define FNC_CROSSSDF
float crossSDF(in float2 st, in float s) {
    float2 size = float2(.25, s);
    return min(rectSDF(st.xy, size.xy),
               rectSDF(st.xy, size.yx));
}
#endif
