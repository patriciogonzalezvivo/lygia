#include "random.glsl"

/*
author: Inigo Quilez
description: returns 2D/3D value noise in the first channel and in the rest the derivatives. For more details read this nice article http://www.iquilezles.org/www/articles/gradientnoise/gradientnoise.htm
use: noised(<vec2|vec3> space)
options:
    NOISED_QUINTIC_INTERPOLATION: Quintic interpolation on/off. Default is off.
license: |
    Copyright © 2017 Inigo Quilez
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_NOISED
#define FNC_NOISED

// return gradient noise (in x) and its derivatives (in yz)
vec3 noised (in vec2 p) {
  // grid
  vec2 i = floor( p );
  vec2 f = fract( p );

#ifdef NOISED_QUINTIC_INTERPOLATION
  // quintic interpolation
  vec2 u = f * f * f * (f * (f * 6. - 15.) + 10.);
  vec2 du = 30. * f * f * (f * (f - 2.) + 1.);
#else
  // cubic interpolation
  vec2 u = f * f * (3. - 2. * f);
  vec2 du = 6. * f * (1. - f);
#endif

  vec2 ga = random2(i + vec2(0., 0.));
  vec2 gb = random2(i + vec2(1., 0.));
  vec2 gc = random2(i + vec2(0., 1.));
  vec2 gd = random2(i + vec2(1., 1.));

  float va = dot(ga, f - vec2(0., 0.));
  float vb = dot(gb, f - vec2(1., 0.));
  float vc = dot(gc, f - vec2(0., 1.));
  float vd = dot(gd, f - vec2(1., 1.));

  return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
               ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
               du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va));
}

// returns 3D value noise (in .x)  and its derivatives (in .yzw)
// https://www.shadertoy.com/view/4dffRH
vec4 noised (in vec3 pos) {
  // grid
  vec3 p = floor(pos);
  vec3 w = fract(pos);

#ifdef NOISED_QUINTIC_INTERPOLATION
  // quintic interpolant
  vec3 u = w * w * w * ( w * (w * 6. - 15.) + 10. );
  vec3 du = 30.0 * w * w * ( w * (w - 2.) + 1.);
#else
  // cubic interpolant
  vec3 u = w * w * (3. - 2. * w);
  vec3 du = 6. * w * (1. - w);
#endif

  // gradients
  vec3 ga = random3(p + vec3(0., 0., 0.));
  vec3 gb = random3(p + vec3(1., 0., 0.));
  vec3 gc = random3(p + vec3(0., 1., 0.));
  vec3 gd = random3(p + vec3(1., 1., 0.));
  vec3 ge = random3(p + vec3(0., 0., 1.));
  vec3 gf = random3(p + vec3(1., 0., 1.));
  vec3 gg = random3(p + vec3(0., 1., 1.));
  vec3 gh = random3(p + vec3(1., 1., 1.));

  // projections
  float va = dot(ga, w - vec3(0., 0., 0.));
  float vb = dot(gb, w - vec3(1., 0., 0.));
  float vc = dot(gc, w - vec3(0., 1., 0.));
  float vd = dot(gd, w - vec3(1., 1., 0.));
  float ve = dot(ge, w - vec3(0., 0., 1.));
  float vf = dot(gf, w - vec3(1., 0., 1.));
  float vg = dot(gg, w - vec3(0., 1., 1.));
  float vh = dot(gh, w - vec3(1., 1., 1.));

  // interpolations
  return vec4( va + u.x*(vb-va) + u.y*(vc-va) + u.z*(ve-va) + u.x*u.y*(va-vb-vc+vd) + u.y*u.z*(va-vc-ve+vg) + u.z*u.x*(va-vb-ve+vf) + (-va+vb+vc-vd+ve-vf-vg+vh)*u.x*u.y*u.z,    // value
               ga + u.x*(gb-ga) + u.y*(gc-ga) + u.z*(ge-ga) + u.x*u.y*(ga-gb-gc+gd) + u.y*u.z*(ga-gc-ge+gg) + u.z*u.x*(ga-gb-ge+gf) + (-ga+gb+gc-gd+ge-gf-gg+gh)*u.x*u.y*u.z +   // derivatives
               du * (vec3(vb,vc,ve) - va + u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) ));
}

#endif
