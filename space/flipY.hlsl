/*
original_author: Patricio Gonzalez Vivo
description: Flip Y axis
use: <float2> flipY(<float2> st)
*/

#ifndef FNC_FLIPY
#define FNC_FLIPY
float2 flipY(in float2 st) {
  return float2(st.x, 1. - st.y);
}
#endif
