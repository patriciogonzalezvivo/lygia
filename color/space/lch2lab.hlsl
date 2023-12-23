/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lch ot Lab color space. 
    Note: LCh is simply Lab but converted to polar coordinates (in degrees).
use: <float3|float4> lch2lab(<float3|float4> color)
*/

#ifndef FNC_LCH2LAB
#define FNC_LCH2LAB
float3 lch2lab(float3 lch) {
    return float3(
        lch.x,
        lch.y * cos(lch.z * 0.01745329251),
        lch.y * sin(lch.z * 0.01745329251)
    );
}
float4 lch2lab(float4 lch) { return float4(lch2lab(lch.xyz),lch.a);}
#endif