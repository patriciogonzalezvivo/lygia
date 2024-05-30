#include "srandom.glsl"

/*
contributors: Inigo Quilez
description: Returns 2D/3D value noise in the first channel and in the rest the derivatives. For more details read this nice article http://www.iquilezles.org/www/articles/gradientnoise/gradientnoise.htm
use: noised(<vec2|vec3> space)
options:
    NOISED_QUINTIC_INTERPOLATION: Quintic interpolation on/off. Default is off.
examples:
    - /shaders/generative_noised.frag
*/

#ifndef NOISED_RANDOM2_FNC
#define NOISED_RANDOM2_FNC srandom2
#endif

#ifndef NOISED_RANDOM3_FNC
#define NOISED_RANDOM3_FNC srandom3
#endif

#ifndef FNC_NOISED
#define FNC_NOISED

// return gradient noise (in x) and its derivatives (in yz)
vec3 noised (in vec2 p) {
    // grid
    vec2 i = floor( p );
    vec2 f = fract( p );

    // quintic interpolation
    vec2 u = f * f * f * (f * (f * 6. - 15.) + 10.);
    vec2 du = 30. * f * f * (f * (f - 2.) + 1.);

    vec2 ga = NOISED_RANDOM2_FNC(i + vec2(0., 0.));
    vec2 gb = NOISED_RANDOM2_FNC(i + vec2(1., 0.));
    vec2 gc = NOISED_RANDOM2_FNC(i + vec2(0., 1.));
    vec2 gd = NOISED_RANDOM2_FNC(i + vec2(1., 1.));

    float va = dot(ga, f - vec2(0., 0.));
    float vb = dot(gb, f - vec2(1., 0.));
    float vc = dot(gc, f - vec2(0., 1.));
    float vd = dot(gd, f - vec2(1., 1.));

    return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
                ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
                du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va));
}

vec4 noised (in vec3 pos) {
    // grid
    vec3 p = floor(pos);
    vec3 w = fract(pos);

    // quintic interpolant
    vec3 u = w * w * w * ( w * (w * 6. - 15.) + 10. );
    vec3 du = 30.0 * w * w * ( w * (w - 2.) + 1.);

    // gradients
    vec3 ga = NOISED_RANDOM3_FNC(p + vec3(0., 0., 0.));
    vec3 gb = NOISED_RANDOM3_FNC(p + vec3(1., 0., 0.));
    vec3 gc = NOISED_RANDOM3_FNC(p + vec3(0., 1., 0.));
    vec3 gd = NOISED_RANDOM3_FNC(p + vec3(1., 1., 0.));
    vec3 ge = NOISED_RANDOM3_FNC(p + vec3(0., 0., 1.));
    vec3 gf = NOISED_RANDOM3_FNC(p + vec3(1., 0., 1.));
    vec3 gg = NOISED_RANDOM3_FNC(p + vec3(0., 1., 1.));
    vec3 gh = NOISED_RANDOM3_FNC(p + vec3(1., 1., 1.));

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
