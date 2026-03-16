#include "../math/const.wgsl"
#include "random.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Voronoi positions and distance to centroids
use: <vec3> voronoi(<vec2> pos, <float> time)
options:
  VORONOI_RANDOM_FNC: nan
examples:
    - /shaders/generative_voronoi.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define VORONOI_RANDOM_FNC(UV) ( 0.5 + 0.5 * sin(time + TAU * random2(UV) ) );

fn voronoi2(uv: vec2f, time: f32) -> vec3f {
    let i_uv = floor(uv);
    let f_uv = fract(uv);
    let rta = vec3f(0.0, 0.0, 10.0);
    for (int j=-1; j<=1; j++ ) {
        for (int i=-1; i<=1; i++ ) {
            let neighbor = vec2f(float(i),float(j));
            let point = VORONOI_RANDOM_FNC(i_uv + neighbor);
            point = 0.5 + 0.5 * sin(time + TAU * point);
            let diff = neighbor + point - f_uv;
            let dist = length(diff);
            if ( dist < rta.z ) {
                rta.xy = point;
                rta.z = dist;
            }
        }
    }
    return rta;
}

fn voronoi2a(p: vec2f) -> vec3f { return voronoi(p, 0.0); }
fn voronoi3(p: vec3f) -> vec3f { return voronoi(p.xy, p.z); }
