#include "../math/mod289.hlsl"
#include "../math/permute.hlsl"
#include "../math/taylorInvSqrt.hlsl"
#include "../math/quintic.hlsl"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: Classic Perlin Noise https://github.com/stegu/webgl-noise
use: cnoise(<float2|float3|float4> pos)
license: |
    Copyright 2021-2023 by Stefan Gustavson and Ian McEwan.
    Published under the terms of the MIT license:
    https://opensource.org/license/mit/
*/

#ifndef FNC_CNOISE
#define FNC_CNOISE

float cnoise(in float2 P) {
    float4 Pi = floor(P.xyxy) + float4(0.0, 0.0, 1.0, 1.0);
    float4 Pf = frac(P.xyxy) - float4(0.0, 0.0, 1.0, 1.0);
    Pi = mod289(Pi); // To avoid truncation effects in permutation
    float4 ix = Pi.xzxz;
    float4 iy = Pi.yyww;
    float4 fx = Pf.xzxz;
    float4 fy = Pf.yyww;

    float4 i = permute(permute(ix) + iy);

    float4 gx = frac(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
    float4 gy = abs(gx) - 0.5 ;
    float4 tx = floor(gx + 0.5);
    gx = gx - tx;

    float2 g00 = float2(gx.x,gy.x);
    float2 g10 = float2(gx.y,gy.y);
    float2 g01 = float2(gx.z,gy.z);
    float2 g11 = float2(gx.w,gy.w);

    float4 norm = taylorInvSqrt(float4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.w;

    float n00 = dot(g00, float2(fx.x, fy.x));
    float n10 = dot(g10, float2(fx.y, fy.y));
    float n01 = dot(g01, float2(fx.z, fy.z));
    float n11 = dot(g11, float2(fx.w, fy.w));

    float2 fade_xy = quintic(Pf.xy);
    float2 n_x = lerp(float2(n00, n01), float2(n10, n11), fade_xy.x);
    float n_xy = lerp(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

float cnoise(in float3 P) {
    float3 Pi0 = floor(P); // Integer part for indexing
    float3 Pi1 = Pi0 + float3(1.0, 1.0, 1.0); // Integer part + 1
    Pi0 = mod289(Pi0);
    Pi1 = mod289(Pi1);
    float3 Pf0 = frac(P); // Fractional part for interpolation
    float3 Pf1 = Pf0 - float3(1.0, 1.0, 1.0); // Fractional part - 1.0
    float4 ix = float4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    float4 iy = float4(Pi0.yy, Pi1.yy);
    float4 iz0 = Pi0.zzzz;
    float4 iz1 = Pi1.zzzz;

    float4 ixy = permute(permute(ix) + iy);
    float4 ixy0 = permute(ixy + iz0);
    float4 ixy1 = permute(ixy + iz1);

    float4 gx0 = ixy0 * (1.0 / 7.0);
    float4 gy0 = frac(floor(gx0) * (1.0 / 7.0)) - 0.5;
    gx0 = frac(gx0);
    float4 gz0 = float4(0.5, 0.5, 0.5, 0.5) - abs(gx0) - abs(gy0);
    float4 sz0 = step(gz0, float4(0.0, 0.0, 0.0, 0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);

    float4 gx1 = ixy1 * (1.0 / 7.0);
    float4 gy1 = frac(floor(gx1) * (1.0 / 7.0)) - 0.5;
    gx1 = frac(gx1);
    float4 gz1 = float4(0.5, 0.5, 0.5, 0.5) - abs(gx1) - abs(gy1);
    float4 sz1 = step(gz1, float4(0.0, 0.0, 0.0, 0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);

    float3 g000 = float3(gx0.x,gy0.x,gz0.x);
    float3 g100 = float3(gx0.y,gy0.y,gz0.y);
    float3 g010 = float3(gx0.z,gy0.z,gz0.z);
    float3 g110 = float3(gx0.w,gy0.w,gz0.w);
    float3 g001 = float3(gx1.x,gy1.x,gz1.x);
    float3 g101 = float3(gx1.y,gy1.y,gz1.y);
    float3 g011 = float3(gx1.z,gy1.z,gz1.z);
    float3 g111 = float3(gx1.w,gy1.w,gz1.w);

    float4 norm0 = taylorInvSqrt(float4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;
    float4 norm1 = taylorInvSqrt(float4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x;
    g011 *= norm1.y;
    g101 *= norm1.z;
    g111 *= norm1.w;

    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, float3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, float3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, float3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, float3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, float3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, float3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);

    float3 fade_xyz = quintic(Pf0);
    float4 n_z = lerp(float4(n000, n100, n010, n110), float4(n001, n101, n011, n111), fade_xyz.z);
    float2 n_yz = lerp(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = lerp(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
}

float cnoise(in float4 P) {
    float4 Pi0 = floor(P); // Integer part for indexing
    float4 Pi1 = Pi0 + 1.0; // Integer part + 1
    Pi0 = mod289(Pi0);
    Pi1 = mod289(Pi1);
    float4 Pf0 = frac(P); // Fractional part for interpolation
    float4 Pf1 = Pf0 - 1.0; // Fractional part - 1.0
    float4 ix = float4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    float4 iy = float4(Pi0.yy, Pi1.yy);
    float4 iz0 = float4(Pi0.zzzz);
    float4 iz1 = float4(Pi1.zzzz);
    float4 iw0 = float4(Pi0.wwww);
    float4 iw1 = float4(Pi1.wwww);

    float4 ixy = permute(permute(ix) + iy);
    float4 ixy0 = permute(ixy + iz0);
    float4 ixy1 = permute(ixy + iz1);
    float4 ixy00 = permute(ixy0 + iw0);
    float4 ixy01 = permute(ixy0 + iw1);
    float4 ixy10 = permute(ixy1 + iw0);
    float4 ixy11 = permute(ixy1 + iw1);

    float4 gx00 = ixy00 * (1.0 / 7.0);
    float4 gy00 = floor(gx00) * (1.0 / 7.0);
    float4 gz00 = floor(gy00) * (1.0 / 6.0);
    gx00 = frac(gx00) - 0.5;
    gy00 = frac(gy00) - 0.5;
    gz00 = frac(gz00) - 0.5;
    float4 gw00 = float4(0.75, 0.75, 0.75, 0.75) - abs(gx00) - abs(gy00) - abs(gz00);
    float4 sw00 = step(gw00, float4(0.0, 0.0, 0.0, 0.0));
    gx00 -= sw00 * (step(0.0, gx00) - 0.5);
    gy00 -= sw00 * (step(0.0, gy00) - 0.5);

    float4 gx01 = ixy01 * (1.0 / 7.0);
    float4 gy01 = floor(gx01) * (1.0 / 7.0);
    float4 gz01 = floor(gy01) * (1.0 / 6.0);
    gx01 = frac(gx01) - 0.5;
    gy01 = frac(gy01) - 0.5;
    gz01 = frac(gz01) - 0.5;
    float4 gw01 = float4(0.75, 0.75, 0.75, 0.75) - abs(gx01) - abs(gy01) - abs(gz01);
    float4 sw01 = step(gw01, float4(0.0, 0.0, 0.0, 0.0));
    gx01 -= sw01 * (step(0.0, gx01) - 0.5);
    gy01 -= sw01 * (step(0.0, gy01) - 0.5);

    float4 gx10 = ixy10 * (1.0 / 7.0);
    float4 gy10 = floor(gx10) * (1.0 / 7.0);
    float4 gz10 = floor(gy10) * (1.0 / 6.0);
    gx10 = frac(gx10) - 0.5;
    gy10 = frac(gy10) - 0.5;
    gz10 = frac(gz10) - 0.5;
    float4 gw10 = float4(0.75, 0.75, 0.75, 0.75) - abs(gx10) - abs(gy10) - abs(gz10);
    float4 sw10 = step(gw10, float4(0.0, 0.0, 0.0, 0.0));
    gx10 -= sw10 * (step(0.0, gx10) - 0.5);
    gy10 -= sw10 * (step(0.0, gy10) - 0.5);

    float4 gx11 = ixy11 * (1.0 / 7.0);
    float4 gy11 = floor(gx11) * (1.0 / 7.0);
    float4 gz11 = floor(gy11) * (1.0 / 6.0);
    gx11 = frac(gx11) - 0.5;
    gy11 = frac(gy11) - 0.5;
    gz11 = frac(gz11) - 0.5;
    float4 gw11 = float4(0.75, 0.75, 0.75, 0.75) - abs(gx11) - abs(gy11) - abs(gz11);
    float4 sw11 = step(gw11, float4(0.0, 0.0, 0.0, 0.0));
    gx11 -= sw11 * (step(0.0, gx11) - 0.5);
    gy11 -= sw11 * (step(0.0, gy11) - 0.5);

    float4 g0000 = float4(gx00.x,gy00.x,gz00.x,gw00.x);
    float4 g1000 = float4(gx00.y,gy00.y,gz00.y,gw00.y);
    float4 g0100 = float4(gx00.z,gy00.z,gz00.z,gw00.z);
    float4 g1100 = float4(gx00.w,gy00.w,gz00.w,gw00.w);
    float4 g0010 = float4(gx10.x,gy10.x,gz10.x,gw10.x);
    float4 g1010 = float4(gx10.y,gy10.y,gz10.y,gw10.y);
    float4 g0110 = float4(gx10.z,gy10.z,gz10.z,gw10.z);
    float4 g1110 = float4(gx10.w,gy10.w,gz10.w,gw10.w);
    float4 g0001 = float4(gx01.x,gy01.x,gz01.x,gw01.x);
    float4 g1001 = float4(gx01.y,gy01.y,gz01.y,gw01.y);
    float4 g0101 = float4(gx01.z,gy01.z,gz01.z,gw01.z);
    float4 g1101 = float4(gx01.w,gy01.w,gz01.w,gw01.w);
    float4 g0011 = float4(gx11.x,gy11.x,gz11.x,gw11.x);
    float4 g1011 = float4(gx11.y,gy11.y,gz11.y,gw11.y);
    float4 g0111 = float4(gx11.z,gy11.z,gz11.z,gw11.z);
    float4 g1111 = float4(gx11.w,gy11.w,gz11.w,gw11.w);

    float4 norm00 = taylorInvSqrt(float4(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
    g0000 *= norm00.x;
    g0100 *= norm00.y;
    g1000 *= norm00.z;
    g1100 *= norm00.w;

    float4 norm01 = taylorInvSqrt(float4(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
    g0001 *= norm01.x;
    g0101 *= norm01.y;
    g1001 *= norm01.z;
    g1101 *= norm01.w;

    float4 norm10 = taylorInvSqrt(float4(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
    g0010 *= norm10.x;
    g0110 *= norm10.y;
    g1010 *= norm10.z;
    g1110 *= norm10.w;

    float4 norm11 = taylorInvSqrt(float4(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
    g0011 *= norm11.x;
    g0111 *= norm11.y;
    g1011 *= norm11.z;
    g1111 *= norm11.w;

    float n0000 = dot(g0000, Pf0);
    float n1000 = dot(g1000, float4(Pf1.x, Pf0.yzw));
    float n0100 = dot(g0100, float4(Pf0.x, Pf1.y, Pf0.zw));
    float n1100 = dot(g1100, float4(Pf1.xy, Pf0.zw));
    float n0010 = dot(g0010, float4(Pf0.xy, Pf1.z, Pf0.w));
    float n1010 = dot(g1010, float4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
    float n0110 = dot(g0110, float4(Pf0.x, Pf1.yz, Pf0.w));
    float n1110 = dot(g1110, float4(Pf1.xyz, Pf0.w));
    float n0001 = dot(g0001, float4(Pf0.xyz, Pf1.w));
    float n1001 = dot(g1001, float4(Pf1.x, Pf0.yz, Pf1.w));
    float n0101 = dot(g0101, float4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
    float n1101 = dot(g1101, float4(Pf1.xy, Pf0.z, Pf1.w));
    float n0011 = dot(g0011, float4(Pf0.xy, Pf1.zw));
    float n1011 = dot(g1011, float4(Pf1.x, Pf0.y, Pf1.zw));
    float n0111 = dot(g0111, float4(Pf0.x, Pf1.yzw));
    float n1111 = dot(g1111, Pf1);

    float4 fade_xyzw = quintic(Pf0);
    float4 n_0w = lerp(float4(n0000, n1000, n0100, n1100), float4(n0001, n1001, n0101, n1101), fade_xyzw.w);
    float4 n_1w = lerp(float4(n0010, n1010, n0110, n1110), float4(n0011, n1011, n0111, n1111), fade_xyzw.w);
    float4 n_zw = lerp(n_0w, n_1w, fade_xyzw.z);
    float2 n_yzw = lerp(n_zw.xy, n_zw.zw, fade_xyzw.y);
    float n_xyzw = lerp(n_yzw.x, n_yzw.y, fade_xyzw.x);
    return 2.2 * n_xyzw;
}
#endif
