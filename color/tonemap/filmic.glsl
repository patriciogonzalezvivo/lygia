/*
contributors: [Jim Hejl, Richard Burgess-Dawson ]
description: Haarm-Peter Duikers curve from John Hables presentation "Uncharted 2 HDR Lighting", Page 140, http://www.gdcvault.com/play/1012459/Uncharted_2__HDR_Lighting
use: <vec3|vec4> tonemapFilmic(<vec3|vec4> x)
*/

#ifndef FNC_TONEMAPFILMIC
#define FNC_TONEMAPFILMIC
vec3 tonemapFilmic(vec3 v) {
    v = max(vec3(0.0), v - 0.004);
    v = (v * (6.2 * v + 0.5)) / (v * (6.2 * v + 1.7) + 0.06);
    return v;
}

vec4 tonemapFilmic(const vec4 x) { return vec4( tonemapFilmic(x.rgb), x.a ); }
#endif