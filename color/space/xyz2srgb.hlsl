/*
contributors: Ronald van Wijnen (@OneDayOfCrypto)
description: Converts a XYZ color to sRGB color space.
use: xyz2rgb(<float3|float4> color)
*/

#ifndef MAT_XYZ2SRGB
#define MAT_XYZ2SRGB
#ifdef CIE_D50
const float3x3 XYZ2SRGB = float3x3(
     3.1338561,-0.9787684, 0.0719453,
    -1.6168667, 1.9161415,-0.2289914,
    -0.4906146, 0.0334540, 1.4052427
);
#else
// CIE D65
const float3x3 XYZ2SRGB = float3x3(
     3.2404542,-0.9692660, 0.0556434,
    -1.5371385, 1.8760108,-0.2040259,
    -0.4985314, 0.0415560, 1.0572252
);
#endif
#endif

#ifndef FNC_XYZ2SRGB
#define FNC_XYZ2SRGB
float3 xyz2srgb(float3 xyz) { return mul(XYZ2SRGB * xyz); }
float4 xyz2srgb(in float4 xyz) { return float4(xyz2srgb(xyz.rgb), xyz.a); }
#endif

