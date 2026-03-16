/*
contributors: [Erik Reinhard, Michael Stark, Peter Shirley, James Ferwerda]
description: Photographic Tone Reproduction for Digital Images. http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf
use: <vec3|vec4> tonemapReinhardJodie(<vec3|vec4> x)
*/

fn tonemapReinhardJodie3(x: vec3f) -> vec3f {
    let l = dot(x, vec3f(0.21250175, 0.71537574, 0.07212251));
    let tc = x / (x + 1.0);
    return mix(x / (l + 1.0), tc, tc); 
}
fn tonemapReinhardJodie4(x: vec4f) -> vec4f { return vec4f( tonemapReinhardJodie(x.rgb), x.a ); }
