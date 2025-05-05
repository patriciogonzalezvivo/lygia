

/*
contributors: [Jim Hejl, Richard Burgess-Dawson ]
description: Haarm-Peter Duikers curve from John Hables presentation "Uncharted 2 HDR Lighting", Page 140, http://www.gdcvault.com/play/1012459/Uncharted_2__HDR_Lighting
use: <vec3|vec4> tonemapFilmic(<vec3|vec4> x)
*/

fn tonemapFilmic3(input_v : vec3f) -> vec3f {
    var v = input_v;
    v = max(vec3(0.0), v - 0.004);                                       
    v = (v * (6.2 * v + 0.5)) / (v * (6.2 * v + 1.7) + 0.06);
    return v;
}

fn tonemapFilmic4(x : vec4f) -> vec4f {
    return vec4(tonemapFilmic3(x.rgb), x.a);
}