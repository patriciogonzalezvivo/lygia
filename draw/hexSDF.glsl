/*\
author: Patricio Gonzalez Vivo
description: Returns a hexagon-shaped SDF
use: hexSDF(<vec2> st)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_HEXSDF
#define FNC_HEXSDF
float hexSDF(in vec2 st) {
    st = abs(st * 2. - 1.);
    return max(abs(st.y), st.x * .866025 + st.y * .5);
}
#endif
