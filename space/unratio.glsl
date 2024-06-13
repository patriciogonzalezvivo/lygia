/*
contributors: Patricio Gonzalez Vivo
description: unFix the aspect ratio
use: ratio(vec2 st, vec2 st_size)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_UNRATIO
#define FNC_UNRATIO
vec2 unratio (in vec2 st, in vec2 s) { return vec2(st.x, st.y*(s.x/s.y)+(s.y*.5-s.x*.5)/s.y); }
#endif