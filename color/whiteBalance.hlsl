#include "space/rgb2yiq.hlsl"
#include "space/yiq2rgb.hlsl"

/*
contributors:
    - Brad Larson
    - Ben Cochran
    - Hugues Lismonde
    - Keitaroh Kobayashi
    - Alaric Cole
    - Matthew Clark
    - Jacob Gundersen
    - Chris Williams
    - Patricio Gonzalez Vivo
description: "Adjust temperature and tint. \nOn mobile does a cheaper algo using Brad\
    \ Larson https://github.com/BradLarson/GPUImage/blob/master/framework/Source/GPUImageWhiteBalanceFilter.m\
    \ \nOn non mobile deas a more accurate ajustment using https://docs.unity3d.com/Packages/com.unity.shadergraph@6.9/manual/White-Balance-Node.html\n"
use: <float3|float4> whiteBalance(<float3|float4> rgb, <float> temperature, <float> tint))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WHITEBALANCE
#define FNC_WHITEBALANCE

float3 whiteBalance(in float3 rgb, in float temperature, in float tint) {
#if defined(TARGET_MOBILE)
    const float3 warmFilter = float3(0.93, 0.54, 0.0);

    float3 yiq = rgb2yiq(rgb);
    // adjust tint
	yiq.b = clamp(yiq.b + tint * 0.05226, -0.5226, 0.5226);
	rgb = yiq2rgb(yiq);
    
    // adjusting temperature
	float3 processed = float3(    (rgb.r < 0.5 ? (2.0 * rgb.r * warmFilter.r) : (1.0 - 2.0 * (1.0 - rgb.r) * (1.0 - warmFilter.r))),
                    (rgb.g < 0.5 ? (2.0 * rgb.g * warmFilter.g) : (1.0 - 2.0 * (1.0 - rgb.g) * (1.0 - warmFilter.g))),
                    (rgb.b < 0.5 ? (2.0 * rgb.b * warmFilter.b) : (1.0 - 2.0 * (1.0 - rgb.b) * (1.0 - warmFilter.b))) );

    return lerp(rgb, processed, temperature * 0.5);
                        
#else
    // Get the CIE xy chromaticity of the reference white point.
    // Note: 0.31271 = x value on the D65 white point
    float x = 0.31271 - temperature * (temperature < 0.0 ? 0.1 : 0.05);
    float standardIlluminantY = 2.87 * x - 3.0 * x * x - 0.27509507;
    float y = standardIlluminantY + tint * 0.05;

    // Calculate the coefficients in the LMS space.
    const float3 w1 = float3(0.949237, 1.03542, 1.08728); // D65 white point

    // CIExyToLMS
    float Y = 1.0;
    float X = Y * x / y;
    float Z = Y * (1.0 - x - y) / y;

    float L =  0.7328 * X + 0.4296 * Y - 0.1624 * Z;
    float M = -0.7036 * X + 1.6975 * Y + 0.0061 * Z;
    float S =  0.0030 * X + 0.0136 * Y + 0.9834 * Z;
    float3 w2 = float3(L, M, S);

    float3 balance = float3(w1.x / w2.x, w1.y / w2.y, w1.z / w2.z);

    // TODO: use our own rgb to lms to rgb
    float3x3 lin2lms_mat = {
        3.90405e-1, 5.49941e-1, 8.92632e-3,
        7.08416e-2, 9.63172e-1, 1.35775e-3,
        2.31082e-2, 1.28021e-1, 9.36245e-1
    };

    float3x3 lms2lin_mat = {
        2.85847e+0, -1.62879e+0, -2.48910e-2,
        -2.10182e-1,  1.15820e+0,  3.24281e-4,
        -4.18120e-2, -1.18169e-1,  1.06867e+0
    };

    float3 lms = mul(lin2lms_mat, rgb);
    lms *= balance;
    return mul(lms2lin_mat, lms);
#endif
}

float4 whiteBalance(in float4 color, in float temperature, in float tint) { return float4( whiteBalance(color.rgb, temperature, tint), color.a); }

#endif