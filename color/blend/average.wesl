/*
contributors: Jamie Owen
description: Photoshop Average blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendAverage(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendAverage(base : f32, blend : f32) -> f32 { return (base + blend) * 0.5; }
fn blendAverage3(base : vec3f, blend : vec3f) -> vec3f { return (base + blend) * 0.5; }
fn blendAverage3Opacity(base : vec3f, blend : vec3f, opacity : f32) -> vec3f { return (blendAverage3(base, blend) * opacity + base * (1.0 - opacity)); }
