/*
author: Patricio Gonzalez Vivo
description: Returns a rectangular SDF
use: rectSDF(<float2> st, <float2> size)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_RECTSDF
#define FNC_RECTSDF
float rectSDF(in float2 st, in float2 s) {
    st = st * 2. - 1.;
    return max( abs(st.x / s.x),
                abs(st.y / s.y) );
}
#endif
