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
    \ \nOn non mobile deas a more accurate adjustment using https://docs.unity3d.com/Packages/com.unity.shadergraph@6.9/manual/White-Balance-Node.html\n"
use: <vec3|vec4> whiteBalance(<vec3|vec4> rgb, <float> temperature, <float> tint))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn whiteBalance3(rgb : vec3f, temp : f32, tint : f32) -> vec3f {
    // Get the CIE xy chromaticity of the reference white point.
    // Note: 0.31271 = x value on the D65 white point

    var x : f32;
    if (temp < 0.0) {
        x = 0.31271 - temp * (0.1);
    } else {
        x = 0.31271 - temp * (0.05);
    }

    let standardIlluminantY = 2.87 * x - 3.0 * x * x - 0.27509507;
    let y = standardIlluminantY + tint * 0.05;

    // CIExyToLMS
    let Y = 1.0;
    let X = Y * x / y;
    let Z = Y * (1.0 - x - y) / y;
    let L = 0.7328 * X + 0.4296 * Y - 0.1624 * Z;
    let M = -0.7036 * X + 1.6975 * Y + 0.0061 * Z;
    let S = 0.0030 * X + 0.0136 * Y + 0.9834 * Z;

    // Calculate the coefficients in the LMS space.
    let w = vec3(0.949237, 1.03542, 1.08728);  // D65 white point
    let balance = w / vec3(L, M, S);

    // TODO: use our own rgb to lms to rgb
    let lin2lms_mat = mat3x3(3.90405e-1, 5.49941e-1, 8.92632e-3, 7.08416e-2, 9.63172e-1, 1.35775e-3, 2.31082e-2,
                             1.28021e-1, 9.36245e-1);

    let lms2lin_mat = mat3x3(2.85847e+0, -1.62879e+0, -2.48910e-2, -2.10182e-1, 1.15820e+0, 3.24281e-4, -4.18120e-2,
                             -1.18169e-1, 1.06867e+0);

    var lms = lin2lms_mat * rgb;
    lms *= balance;
    return lms2lin_mat * lms;
}

fn whiteBalance4(color : vec4f, temp : f32, tint : f32) -> vec4f {
    return vec4(whiteBalance3(color.rgb, temp, tint), color.a);
}