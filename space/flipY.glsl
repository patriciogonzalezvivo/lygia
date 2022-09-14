/*
original_author: Patricio Gonzalez Vivo
description: Flip Y axis
use: <vec2> flipY(<vec2> st)
*/

#ifndef FNC_FLIPY
#define FNC_FLIPY
vec2 flipY(in vec2 st) {
  return vec2(st.x, 1. - st.y);
}
#endif
