#include "space/ryb2rgb.wgsl"
#include "space/rgb2ryb.wgsl"
#include "../math/sum.wgsl"

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

fn mixRYB3(A: vec3f, B: vec3f, p: f32) -> vec3f { return ryb2rgb(mix(rgb2ryb(A),rgb2ryb(B), p)); }
fn mixRYB4(A: vec4f, B: vec4f, p: f32) -> vec4f { return ryb2rgb(mix(rgb2ryb(A),rgb2ryb(B), p)); }

fn mixRYB4a(A: vec4f, B: vec4f, C: vec4f) -> vec4f { return vec4f(ryb2rgb(rgb2ryb(A.rgb) * A.a + rgb2ryb(B.rgb) * B.a + rgb2ryb(C.rgb) * C.a), sum(vec3f(A.a, B.a, C.a))); }
fn mixRYB3a(A: vec3f, B: vec3f, C: vec3f, p: vec3f) -> vec3f { return ryb2rgb(rgb2ryb(A) * p.x + rgb2ryb(B) * p.y + rgb2ryb(C) * p.z ); }

fn mixRYB4b(A: vec4f, B: vec4f, C: vec4f, D: vec4f) -> vec4f { return vec4f(ryb2rgb(rgb2ryb(A.rgb) * A.a + rgb2ryb(B.rgb) * B.a + rgb2ryb(C.rgb) * C.a + rgb2ryb(D.rgb) * D.a), sum(vec4f(A.a, B.a, C.a, D.a))); }
fn mixRYB3b(A: vec3f, B: vec3f, C: vec3f, D: vec3f, p: vec4f) -> vec3f { return ryb2rgb(rgb2ryb(A) * p.x + rgb2ryb(B) * p.y + rgb2ryb(C) * p.z + rgb2ryb(D) * p.w); }
