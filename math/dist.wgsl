/*
contributors: Shadi El Hajj
description: Commonly used distance functions.
notes:
    - While the GLSL and HLSL versions of this file support defining the default distance function (DIST_FNC), WGSL does not have a standard way to do this. As such, the current implementation uses distEuclidean.
options:
    - DIST_MINKOWSKI_P: the power of the Minkowski distance function (1.0 Manhattan, 2.0 Euclidean, Infinity Chebychev)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

const DIST_MINKOWSKI_P: f32 = 2.0; // 1: Manhattan, 2: Euclidean, Infinity: Chebychev

fn distEuclidean2(a: vec2f, b: vec2f) -> f32 { return distance(a, b); }
fn distEuclidean3(a: vec3f, b: vec3f) -> f32 { return distance(a, b); }
fn distEuclidean4(a: vec4f, b: vec4f) -> f32 { return distance(a, b); }

// https://en.wikipedia.org/wiki/Taxicab_geometry
fn distManhattan2(a: vec2f, b: vec2f) -> f32 { return abs(a.x - b.x) + abs(a.y - b.y); }
fn distManhattan3(a: vec3f, b: vec3f) -> f32 { return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z); }
fn distManhattan4(a: vec4f, b: vec4f) -> f32 { return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) + abs(a.w - b.w); }

// https://en.wikipedia.org/wiki/Chebyshev_distance
fn distChebychev2(a: vec2f, b: vec2f) -> f32 { return max(abs(a.x - b.x), abs(a.y - b.y)); }
fn distChebychev3(a: vec3f, b: vec3f) -> f32 { return max(abs(a.x - b.x), max(abs(a.y - b.y), abs(a.z - b.z))); }
fn distChebychev4(a: vec4f, b: vec4f) -> f32 { return max(abs(a.x - b.x), max(abs(a.y - b.y), max(abs(a.z - b.z), abs(a.w - b.w) ))); }

// https://en.wikipedia.org/wiki/Minkowski_distance
fn distMinkowski2(a: vec2f, b: vec2f) -> f32 { return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P) + pow(abs(a.y - b.y), DIST_MINKOWSKI_P), 1.0 / DIST_MINKOWSKI_P); }
fn distMinkowski3(a: vec3f, b: vec3f) -> f32 { return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P) + pow(abs(a.y - b.y), DIST_MINKOWSKI_P) + pow(abs(a.z - b.z), DIST_MINKOWSKI_P), 1.0 / DIST_MINKOWSKI_P); }
fn distMinkowski4(a: vec4f, b: vec4f) -> f32 { return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P) + pow(abs(a.y - b.y), DIST_MINKOWSKI_P) + pow(abs(a.z - b.z), DIST_MINKOWSKI_P) + pow(abs(a.w - b.w), DIST_MINKOWSKI_P), 1.0 / DIST_MINKOWSKI_P); }

fn dist2(a: vec2f, b: vec2f) -> f32 { return distEuclidean2(a, b); }
fn dist3(a: vec3f, b: vec3f) -> f32 { return distEuclidean3(a, b); }
fn dist4(a: vec4f, b: vec4f) -> f32 { return distEuclidean4(a, b); }
