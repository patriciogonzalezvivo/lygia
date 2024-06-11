
#include "space/ryb2rgb.glsl"
#include "space/rgb2ryb.glsl"
#include "../math/sum.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Mix colors in RYB space.
use: 
    - <vec3|vec4> mixRYB(<vec3|vec4> colA, <vec3|vec4> colB, <float> p)
    - <vec4> mixRYB(<vec4> colA, <vec4> colB, <vec4> colC)
    - <vec3> mixRYB(<vec3> colA, <vec3> colB, <vec3> colC, <vec3> p)
    - <vec4> mixRYB(<vec4> colA, <vec4> colB, <vec4> colC, <vec4> colD)
    - <vec3> mixRYB(<vec3> colA, <vec3> colB, <vec3> colC, <vec3> colD, <vec4> p) 
options:
    - RYB_FAST: Use a faster approximation of the RYB space
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_mix_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MIXRYB
#define FNC_MIXRYB

vec3 mixRYB(vec3 A, vec3 B, float p) { return ryb2rgb(mix(rgb2ryb(A),rgb2ryb(B), p)); }
vec4 mixRYB(vec4 A, vec4 B, float p) { return ryb2rgb(mix(rgb2ryb(A),rgb2ryb(B), p)); }

vec4 mixRYB(vec4 A, vec4 B, vec4 C) { return vec4(ryb2rgb(rgb2ryb(A.rgb) * A.a + rgb2ryb(B.rgb) * B.a + rgb2ryb(C.rgb) * C.a), sum(vec3(A.a, B.a, C.a))); }
vec3 mixRYB(vec3 A, vec3 B, vec3 C, vec3 p) { return ryb2rgb(rgb2ryb(A) * p.x + rgb2ryb(B) * p.y + rgb2ryb(C) * p.z ); }

vec4 mixRYB(vec4 A, vec4 B, vec4 C, vec4 D) { return vec4(ryb2rgb(rgb2ryb(A.rgb) * A.a + rgb2ryb(B.rgb) * B.a + rgb2ryb(C.rgb) * C.a + rgb2ryb(D.rgb) * D.a), sum(vec4(A.a, B.a, C.a, D.a))); }
vec3 mixRYB(vec3 A, vec3 B, vec3 C, vec3 D, vec4 p) { return ryb2rgb(rgb2ryb(A) * p.x + rgb2ryb(B) * p.y + rgb2ryb(C) * p.z + rgb2ryb(D) * p.w); }

#endif