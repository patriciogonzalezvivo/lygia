/*
author: Patricio Gonzalez Vivo
description: Returns a flower shaped SDF
use: flowerSDF(<vec2> st, <int> n_sides)
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
    Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
    https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_FLOWERSDF
#define FNC_FLOWERSDF
float flowerSDF(vec2 st, int N) {
    st = st * 2.0 - 1.0;
    float r = length(st) * 2.0;
    float a = atan(st.y, st.x);
    float v = float(N) * 0.5;
    return 1.0 - (abs(cos(a * v)) *  0.5 + 0.5) / r;
}
#endif