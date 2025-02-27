/*
contributors: Jamie Owen
description: Photoshop Add blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendAdd(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendAdd(base : f32, blend : f32) -> f32 { return min(base + blend, 1.0); }
fn blendAdd3(base : vec3f, blend : vec3f) -> vec3f { return min(base + blend, vec3(1.0)); }
fn blendAdd3Opacity(base : vec3f, blend : vec3f, opacity : f32) -> vec3f { return blendAdd3(base, blend) * opacity + base * (1.0 - opacity); }
