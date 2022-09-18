#include "srandom.hlsl"

/*
original_author: Inigo Quilez
description: returns 2D/3D value noise in the first channel and in the rest the derivatives. For more details read this nice article http://www.iquilezles.org/www/articles/gradientnoise/gradientnoise.htm
use: noised(<float2|float3> space)
options:
    NOISED_QUINTIC_INTERPOLATION: Quintic interpolation on/off. Default is off.
license: |
    Copyright © 2017 Inigo Quilez
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_NOISED
#define FNC_NOISED

// return gradient noise (in x) and its derivatives (in yz)
float3 noised (in float2 p) {
  // grid
  float2 i = floor( p );
  float2 f = frac( p );

#ifdef NOISED_QUINTIC_INTERPOLATION
  // quintic interpolation
  float2 u = f * f * f * (f * (f * 6. - 15.) + 10.);
  float2 du = 30. * f * f * (f * (f - 2.) + 1.);
#else
  // cubic interpolation
  float2 u = f * f * (3. - 2. * f);
  float2 du = 6. * f * (1. - f);
#endif

  float2 ga = srandom2(i + float2(0., 0.));
  float2 gb = srandom2(i + float2(1., 0.));
  float2 gc = srandom2(i + float2(0., 1.));
  float2 gd = srandom2(i + float2(1., 1.));

  float va = dot(ga, f - float2(0., 0.));
  float vb = dot(gb, f - float2(1., 0.));
  float vc = dot(gc, f - float2(0., 1.));
  float vd = dot(gd, f - float2(1., 1.));

  return float3(va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
                ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
                du * (u.yx*(va-vb-vc+vd) + float2(vb,vc) - va));
}


float4 noised( in float3 x ) {
    float3 p = floor(x);
    float3 w = frac(x);
    
    float3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    float3 du = 30.0*w*w*(w*(w-2.0)+1.0);

    float a = srandom3( p + float3(0.0, 0.0, 0.0) );
    float b = srandom3( p + float3(1.0, 0.0, 0.0) );
    float c = srandom3( p + float3(0.0, 1.0, 0.0) );
    float d = srandom3( p + float3(1.0, 1.0, 0.0) );
    float e = srandom3( p + float3(0.0, 0.0, 1.0) );
    float f = srandom3( p + float3(1.0, 0.0, 1.0) );
    float g = srandom3( p + float3(0.0, 1.0, 1.0) );
    float h = srandom3( p + float3(1.0, 1.0, 1.0) );

    float k0 =  a;
    float k1 =  b - a;
    float k2 =  c - a;
    float k3 =  e - a;
    float k4 =  a - b - c + d;
    float k5 =  a - c - e + g;
    float k6 =  a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return float4(  -1.0 + 2.0 * (k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z), 
                    2.0* du * float3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                                      k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                                      k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}
#endif
