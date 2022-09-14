/*
original_author: Patricio Gonzalez Vivo
description: unFix the aspect ratio 
use: ratio(vec2 st, vec2 st_size)
*/

#ifndef FNC_UNRATIO
#define FNC_UNRATIO
vec2 unratio (in vec2 st, in vec2 size) {
    return vec2(st.x, st.y*(size.x/size.y)+(size.y*.5-size.x*.5)/size.y);
}
#endif