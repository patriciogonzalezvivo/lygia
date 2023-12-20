/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a sRGB color to XYZ color space.
    Based on http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
use: srgb2xyz(<float3|float4> color)
*/

#ifndef MAT_SRGB2XYZ
#define MAT_SRGB2XYZ
#ifdef CIE_D50
const float3x3 SRGB2XYZ = float3x3( 0.4360747, 0.2225045, 0.0139322,
                                    0.3850649, 0.7168786, 0.0971045,
                                    0.1430804, 0.0606169, 0.7141733);
#else
const float3x3 SRGB2XYZ = float3x3( 0.4124564, 0.2126729, 0.0193339,
                                    0.3575761, 0.7151522, 0.1191920,
                                    0.1804375, 0.0721750, 0.9503041);
#endif
#endif

#ifndef FNC_SRGB2XYZ
#define FNC_SRGB2XYZ
float3 srgb2xyz(float3 srgb) { return mul(SRGB2XYZ, srgb);}
float4 srgb2xyz(float4 srgb) { return float4(srgb2xyz(srgb.rgb),srgb.a); }
#endif