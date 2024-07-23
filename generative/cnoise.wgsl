#include "../math/mod289.wgsl"
#include "../math/permute.wgsl"
#include "../math/taylorInvSqrt.wgsl"
#include "../math/quintic.wgsl"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: Classic Perlin Noise https://github.com/stegu/webgl-noise
use: cnoise2/3/4(<vec2f|vec3f|vec4f> pos)
license: |
    Copyright 2021-2023 by Stefan Gustavson and Ian McEwan.
    Published under the terms of the MIT license:
    https://opensource.org/license/mit/
*/

fn cnoise2(P: vec2f) -> f32 {
  var Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  let Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289_4(Pi); // To avoid truncation effects in permutation
  let ix = Pi.xzxz;
  let iy = Pi.yyww;
  let fx = Pf.xzxz;
  let fy = Pf.yyww;

  let i = permute4(permute4(ix) + iy);

  var gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  let gy = abs(gx) - 0.5 ;
  let tx = floor(gx + 0.5);
  gx = gx - tx;

  var g00 = vec2(gx.x,gy.x);
  var g10 = vec2(gx.y,gy.y);
  var g01 = vec2(gx.z,gy.z);
  var g11 = vec2(gx.w,gy.w);

  let norm = taylorInvSqrt4(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 = g00 * norm.x;
  g01 = g01 * norm.y;
  g10 = g10 * norm.z;
  g11 = g11 * norm.w;

  let n00 = dot(g00, vec2(fx.x, fy.x));
  let n10 = dot(g10, vec2(fx.y, fy.y));
  let n01 = dot(g01, vec2(fx.z, fy.z));
  let n11 = dot(g11, vec2(fx.w, fy.w));

  let fade_xy = quintic2(Pf.xy);
  let n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

fn cnoise3(P: vec3f) -> f32 {
  var Pi0 = floor(P); // Integer part for indexing
  var Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod289_3(Pi0);
  Pi1 = mod289_3(Pi1);
  let Pf0 = fract(P); // Fractional part for interpolation
  let Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  let ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy = vec4(Pi0.yy, Pi1.yy);
  let iz0 = Pi0.zzzz;
  let iz1 = Pi1.zzzz;

  let ixy = permute4(permute4(ix) + iy);
  let ixy0 = permute4(ixy + iz0);
  let ixy1 = permute4(ixy + iz1);

  var gx0 = ixy0 * (1.0 / 7.0);
  var gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  let gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  let sz0 = step(gz0, vec4(0.0));
  gx0 = gx0 - (sz0 * (step(vec4(0.0), gx0) - 0.5));
  gy0 = gy0 - (sz0 * (step(vec4(0.0), gy0) - 0.5));

  var gx1 = ixy1 * (1.0 / 7.0);
  var gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  let gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  let sz1 = step(gz1, vec4(0.0));
  gx1 = gx1 - (sz1 * (step(vec4(0.0), gx1) - 0.5));
  gy1 = gy1 - (sz1 * (step(vec4(0.0), gy1) - 0.5));

  var g000 = vec3(gx0.x,gy0.x,gz0.x);
  var g100 = vec3(gx0.y,gy0.y,gz0.y);
  var g010 = vec3(gx0.z,gy0.z,gz0.z);
  var g110 = vec3(gx0.w,gy0.w,gz0.w);
  var g001 = vec3(gx1.x,gy1.x,gz1.x);
  var g101 = vec3(gx1.y,gy1.y,gz1.y);
  var g011 = vec3(gx1.z,gy1.z,gz1.z);
  var g111 = vec3(gx1.w,gy1.w,gz1.w);

  let norm0 = taylorInvSqrt4(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 = g000 * norm0.x;
  g010 = g010 * norm0.y;
  g100 = g100 * norm0.z;
  g110 = g110 * norm0.w;
  let norm1 = taylorInvSqrt4(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 = g001 * norm1.x;
  g011 = g011 * norm1.y;
  g101 = g101 * norm1.z;
  g111 = g111 * norm1.w;

  let n000 = dot(g000, Pf0);
  let n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  let n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  let n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  let n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  let n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  let n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  let n111 = dot(g111, Pf1);

  let fade_xyz = quintic3(Pf0);
  let n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  let n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

fn cnoise4(P: vec4f) -> f32 {
  var Pi0 = floor(P); // Integer part for indexing
  var Pi1 = Pi0 + 1.0; // Integer part + 1
  Pi0 = mod289_4(Pi0);
  Pi1 = mod289_4(Pi1);
  let Pf0 = fract(P); // Fractional part for interpolation
  let Pf1 = Pf0 - 1.0; // Fractional part - 1.0
  let ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy = vec4(Pi0.yy, Pi1.yy);
  let iz0 = vec4(Pi0.zzzz);
  let iz1 = vec4(Pi1.zzzz);
  let iw0 = vec4(Pi0.wwww);
  let iw1 = vec4(Pi1.wwww);

  let ixy = permute4(permute4(ix) + iy);
  let ixy0 = permute4(ixy + iz0);
  let ixy1 = permute4(ixy + iz1);
  let ixy00 = permute4(ixy0 + iw0);
  let ixy01 = permute4(ixy0 + iw1);
  let ixy10 = permute4(ixy1 + iw0);
  let ixy11 = permute4(ixy1 + iw1);

  var gx00 = ixy00 * (1.0 / 7.0);
  var gy00 = floor(gx00) * (1.0 / 7.0);
  var gz00 = floor(gy00) * (1.0 / 6.0);
  gx00 = fract(gx00) - 0.5;
  gy00 = fract(gy00) - 0.5;
  gz00 = fract(gz00) - 0.5;
  var gw00 = vec4(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
  var sw00 = step(gw00, vec4(0.0));
  gx00 = gx00 - (sw00 * (step(vec4(0.0), gx00) - 0.5));
  gy00 = gy00 - (sw00 * (step(vec4(0.0), gy00) - 0.5));

  var gx01 = ixy01 * (1.0 / 7.0);
  var gy01 = floor(gx01) * (1.0 / 7.0);
  var gz01 = floor(gy01) * (1.0 / 6.0);
  gx01 = fract(gx01) - 0.5;
  gy01 = fract(gy01) - 0.5;
  gz01 = fract(gz01) - 0.5;
  var gw01 = vec4(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
  var sw01 = step(gw01, vec4(0.0));
  gx01 = gx01 - (sw01 * (step(vec4(0.0), gx01) - 0.5));
  gy01 = gy01 - (sw01 * (step(vec4(0.0), gy01) - 0.5));

  var gx10 = ixy10 * (1.0 / 7.0);
  var gy10 = floor(gx10) * (1.0 / 7.0);
  var gz10 = floor(gy10) * (1.0 / 6.0);
  gx10 = fract(gx10) - 0.5;
  gy10 = fract(gy10) - 0.5;
  gz10 = fract(gz10) - 0.5;
  var gw10 = vec4(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
  var sw10 = step(gw10, vec4(0.0));
  gx10 = gx10 - (sw10 * (step(vec4(0.0), gx10) - 0.5));
  gy10 = gy10 - (sw10 * (step(vec4(0.0), gy10) - 0.5));

  var gx11 = ixy11 * (1.0 / 7.0);
  var gy11 = floor(gx11) * (1.0 / 7.0);
  var gz11 = floor(gy11) * (1.0 / 6.0);
  gx11 = fract(gx11) - 0.5;
  gy11 = fract(gy11) - 0.5;
  gz11 = fract(gz11) - 0.5;
  var gw11 = vec4(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
  var sw11 = step(gw11, vec4(0.0));
  gx11 = gx11 - (sw11 * (step(vec4(0.0), gx11) - 0.5));
  gy11 = gy11 - (sw11 * (step(vec4(0.0), gy11) - 0.5));

  var g0000 = vec4(gx00.x,gy00.x,gz00.x,gw00.x);
  var g1000 = vec4(gx00.y,gy00.y,gz00.y,gw00.y);
  var g0100 = vec4(gx00.z,gy00.z,gz00.z,gw00.z);
  var g1100 = vec4(gx00.w,gy00.w,gz00.w,gw00.w);
  var g0010 = vec4(gx10.x,gy10.x,gz10.x,gw10.x);
  var g1010 = vec4(gx10.y,gy10.y,gz10.y,gw10.y);
  var g0110 = vec4(gx10.z,gy10.z,gz10.z,gw10.z);
  var g1110 = vec4(gx10.w,gy10.w,gz10.w,gw10.w);
  var g0001 = vec4(gx01.x,gy01.x,gz01.x,gw01.x);
  var g1001 = vec4(gx01.y,gy01.y,gz01.y,gw01.y);
  var g0101 = vec4(gx01.z,gy01.z,gz01.z,gw01.z);
  var g1101 = vec4(gx01.w,gy01.w,gz01.w,gw01.w);
  var g0011 = vec4(gx11.x,gy11.x,gz11.x,gw11.x);
  var g1011 = vec4(gx11.y,gy11.y,gz11.y,gw11.y);
  var g0111 = vec4(gx11.z,gy11.z,gz11.z,gw11.z);
  var g1111 = vec4(gx11.w,gy11.w,gz11.w,gw11.w);

  let norm00 = taylorInvSqrt4(vec4(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 = g0000 * norm00.x;
  g0100 = g0100 * norm00.y;
  g1000 = g1000 * norm00.z;
  g1100 = g1100 * norm00.w;

  let norm01 = taylorInvSqrt4(vec4(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 = g0001 * norm01.x;
  g0101 = g0101 * norm01.y;
  g1001 = g1001 * norm01.z;
  g1101 = g1101 * norm01.w;

  let norm10 = taylorInvSqrt4(vec4(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 = g0010 * norm10.x;
  g0110 = g0110 * norm10.y;
  g1010 = g1010 * norm10.z;
  g1110 = g1110 * norm10.w;

  let norm11 = taylorInvSqrt4(vec4(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 = g0011 * norm11.x;
  g0111 = g0111 * norm11.y;
  g1011 = g1011 * norm11.z;
  g1111 = g1111 * norm11.w;

  let n0000 = dot(g0000, Pf0);
  let n1000 = dot(g1000, vec4(Pf1.x, Pf0.yzw));
  let n0100 = dot(g0100, vec4(Pf0.x, Pf1.y, Pf0.zw));
  let n1100 = dot(g1100, vec4(Pf1.xy, Pf0.zw));
  let n0010 = dot(g0010, vec4(Pf0.xy, Pf1.z, Pf0.w));
  let n1010 = dot(g1010, vec4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  let n0110 = dot(g0110, vec4(Pf0.x, Pf1.yz, Pf0.w));
  let n1110 = dot(g1110, vec4(Pf1.xyz, Pf0.w));
  let n0001 = dot(g0001, vec4(Pf0.xyz, Pf1.w));
  let n1001 = dot(g1001, vec4(Pf1.x, Pf0.yz, Pf1.w));
  let n0101 = dot(g0101, vec4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  let n1101 = dot(g1101, vec4(Pf1.xy, Pf0.z, Pf1.w));
  let n0011 = dot(g0011, vec4(Pf0.xy, Pf1.zw));
  let n1011 = dot(g1011, vec4(Pf1.x, Pf0.y, Pf1.zw));
  let n0111 = dot(g0111, vec4(Pf0.x, Pf1.yzw));
  let n1111 = dot(g1111, Pf1);

  let fade_xyzw = quintic4(Pf0);
  let n_0w = mix(vec4(n0000, n1000, n0100, n1100), vec4(n0001, n1001, n0101, n1101), fade_xyzw.w);
  let n_1w = mix(vec4(n0010, n1010, n0110, n1110), vec4(n0011, n1011, n0111, n1111), fade_xyzw.w);
  let n_zw = mix(n_0w, n_1w, fade_xyzw.z);
  let n_yzw = mix(n_zw.xy, n_zw.zw, fade_xyzw.y);
  let n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
  return 2.2 * n_xyzw;
}
