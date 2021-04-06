/*
author: Patricio Gonzalez Vivo
description: Returns a triangle-shaped sdf
use: triSDF(<vec2> st)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_TRISDF
#define FNC_TRISDF
float triSDF(in vec2 st) {
    st = (st * 2. - 1.) * 2.;
    return max(abs(st.x) * .866025 + st.y * .5, -st.y * .5);
}
#endif
