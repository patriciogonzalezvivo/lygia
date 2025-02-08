/*
contributors: Patricio Gonzalez Vivo
description: "Converts a Lch to Lab color space. \nNote: LCh is simply Lab but converted to polar coordinates (in degrees).\n"
use: <vec3|vec4> lch2lab(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LCH2LAB
#define FNC_LCH2LAB
vec3 lch2lab(vec3 lch) {
    return vec3(
        lch.x,
        lch.y * cos(lch.z * 0.01745329251),
        lch.y * sin(lch.z * 0.01745329251)
    );
}
vec4 lch2lab(vec4 lch) { return vec4(lch2lab(lch.xyz),lch.a);}
#endif