#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Marbling: displace a texture through an accumulated inverse map instead of
    re-distorting it every frame. The map stores, for every pixel, where its colour
    must be read from. Stored in 16-bit fixed point (two 8-bit channels per axis) so
    it survives 8-bit buffers; the decode is linear in the channels, so bilinear
    filtering of the packed texels interpolates the map correctly.

    Typical ping-pong use: each frame carry the map back along a velocity field
    (map(st) = map'(st - v * pixel)), and read the ink once through it with marble().
    Periodically bake the result into the ink and reset the map to identity, so the
    colour is resampled once per cycle rather than once per frame (no blur build up).

    A cycle is `period` frames long: marbleReset() is true on its first frame (the map
    restarts from identity) and marbleBake() on its last (bake the displaced ink and
    stamp the live scene). On any other frame, read the ink through the map.
use:
    - <vec4> marbleEncode(<vec2> uv)
    - <vec2> marbleDecode(<vec4> encoded)
    - <vec4> marble(<SAMPLER_TYPE> inkTex, <SAMPLER_TYPE> mapTex, <vec2> st)
    - <bool> marbleReset(<float> frame, <float> period)
    - <bool> marbleBake(<float> frame, <float> period)
    - <bool> marbleReset(<float> frame) // period defaults to MARBLE_PERIOD
    - <bool> marbleBake(<float> frame)  // period defaults to MARBLE_PERIOD
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - MARBLE_PERIOD: frames between ink bakes, defaults to 60.0
    - MARBLE_FADE: ink kept per bake, defaults to 0.985
    - MARBLE_FLOW_SPEED: how fast the ink follows the fluid, defaults to 1.0 (0.1 = the speed before the units fix)
    - MARBLE_SAMPLER_FNC(TEX, UV): function used to sample both the ink and the map, defaults to SAMPLER_FNC(TEX, UV). Use a clamp-to-edge sampler to avoid wrapping at the borders
examples:
    - /shaders/splat_strokes_disolve
    - /shaders/splat_strokes_solvent
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MARBLE_PERIOD
#define MARBLE_PERIOD   60.0    // frames between ink bakes
#endif

#ifndef MARBLE_FADE
#define MARBLE_FADE     0.985   // ink kept per bake
#endif

#ifndef MARBLE_FLOW_SPEED
#define MARBLE_FLOW_SPEED 1.0   // how fast the ink follows the fluid (1.0 = exactly with it; 0.1 = the speed before the units fix)
#endif

#ifndef MARBLE_SAMPLER_FNC
#define MARBLE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_MARBLE
#define FNC_MARBLE

// Pack a uv in [-0.5, 1.5] into RGBA (hi/lo bytes for x in R,G and y in B,A)
vec4 marbleEncode(vec2 uv) {
    vec2 m = clamp((uv + 0.5) * 0.5, 0.0, 1.0) * 255.0;
    vec2 hi = floor(m);
    return vec4(hi.x / 255.0, m.x - hi.x, hi.y / 255.0, m.y - hi.y);
}

// Inverse of marbleEncode()
vec2 marbleDecode(vec4 c) {
    return (c.rb * 255.0 + c.ga) / 255.0 * 2.0 - 0.5;
}

// Read the ink where the map says this pixel came from
vec4 marble(SAMPLER_TYPE inkTex, SAMPLER_TYPE mapTex, vec2 st) {
    return MARBLE_SAMPLER_FNC(inkTex, marbleDecode(MARBLE_SAMPLER_FNC(mapTex, st)));
}


// First frame of a cycle (and frame 0): the map restarts from identity
bool marbleReset(float frame, float period) {
    return frame < 1.0 || mod(frame, period) <= 0.5;
}

// Last frame of a cycle (and frame 0): bake the displaced ink into the ink buffer
bool marbleBake(float frame, float period) {
    return frame < 1.0 || mod(frame, period) > period - 1.5;
}

bool marbleReset(float frame) { return marbleReset(frame, MARBLE_PERIOD); }
bool marbleBake(float frame)  { return marbleBake(frame, MARBLE_PERIOD); }

#endif
