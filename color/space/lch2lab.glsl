/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lch ot Lab color space. 
    Note: LCh is simply Lab but converted to polar coordinates (in degrees).
use: <vec3|vec4> lch2lab(<vec3|vec4> color)
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