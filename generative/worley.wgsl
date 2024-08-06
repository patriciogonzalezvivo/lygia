#include "random.wgsl"
#include "../math/dist.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Worley noise. Returns vec2(F1, F2)
use: <vec2> worley2(<vec2|vec3> pos)
notes:
    - While the GLSL and HLSL versions of this file support other distance functions, WGSL does not have a standard way to do this. As such, the current implementation uses distEuclidean.
options:
    - WORLEY_JITTER: amount of pattern randomness. With 1.0 being the default and 0.0 resulting in a perfectly symmetrical pattern.
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/generative_worley.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const WORLEY_JITTER: f32 = 1.0;

fn worley22(p: vec2f) -> vec2f {
    let n = floor( p );
    let f = fract( p );

    var distF1 = 1.0;
    var distF2 = 1.0;
    var off1 = vec2(0.0); 
    var pos1 = vec2(0.0);
    var off2 = vec2(0.0);
    var pos2 = vec2(0.0);
    for(var j = -1; j <= 1; j++) {
        for(var i = -1; i <= 1; i++) {	
            let  g = vec2(f32(i), f32(j));
            let  o = random22( n + g ) * WORLEY_JITTER;
            let  p = g + o;
            let d = distEuclidean2(p, f);
            if (d < distF1) {
                distF2 = distF1;
                distF1 = d;
                off2 = off1;
                off1 = g;
                pos2 = pos1;
                pos1 = p;
            }
            else if (d < distF2) {
                distF2 = d;
                off2 = g;
                pos2 = p;
            }
        }
    }

    return vec2(distF1, distF2);
}

fn worley2(p: vec2f) -> f32 { return 1.0-worley22(p).x; }

fn worley32(p: vec3f) -> vec2f {
    let n = floor( p );
    let f = fract( p );

    var distF1 = 1.0;
    var distF2 = 1.0;
    var off1 = vec3(0.0);
    var pos1 = vec3(0.0);
    var off2 = vec3(0.0);
    var pos2 = vec3(0.0);
    for(var k = -1; k <= 1; k++) {
        for(var j = -1; j <= 1; j++) {
            for(var i=-1; i <= 1; i++) {	
                let  g = vec3(f32(i), f32(j), f32(k));
                let  o = random33( n + g ) * WORLEY_JITTER;
                let  p = g + o;
                let d = distEuclidean3(p, f);
                if (d < distF1) {
                    distF2 = distF1;
                    distF1 = d;
                    off2 = off1;
                    off1 = g;
                    pos2 = pos1;
                    pos1 = p;
                }
                else if (d < distF2) {
                    distF2 = d;
                    off2 = g;
                    pos2 = p;
                }
            }
        }
    }

    return vec2(distF1, distF2);
}

fn worley3(p: vec3f) -> f32 { return 1.0-worley32(p).x; }
