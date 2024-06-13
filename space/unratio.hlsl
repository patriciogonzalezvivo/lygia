/*
contributors: Patricio Gonzalez Vivo
description: unFix the aspect ratio
use: ratio(float2 st, float2 st_size)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_UNRATIO
#define FNC_UNRATIO
float2 unratio (in float2 st, in float2 size) {
    return float2(st.x, st.y*(size.x/size.y)+(size.y*.5-size.x*.5)/size.y);
}
#endif