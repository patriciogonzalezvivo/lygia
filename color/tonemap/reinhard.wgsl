/*
contributors: [Erik Reinhard, Michael Stark, Peter Shirley, James Ferwerda]
description: Photographic Tone Reproduction for Digital Images. http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf
use: <vec3|vec4> tonemapReinhard(<vec3|vec4> x)
*/

fn tonemapReinhard3(v: vec3f) -> vec3f { return v / (1.0 + dot(v, vec3f(0.21250175, 0.71537574, 0.07212251))); }
fn tonemapReinhard4(v: vec4f) -> vec4f { return vec4f( tonemapReinhard(v.rgb), v.a ); }
