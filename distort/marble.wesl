@if(MARBLE_PERIOD)
import constants::MARBLE_PERIOD;

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

    WESL conditionals:
    - @if(MARBLE_PERIOD): uses constants::MARBLE_PERIOD (f32, frames between ink bakes) instead of the default 60.0

    Textures are sampled at mip level 0 so the functions can be called from non-uniform
    control flow. The sampler's address mode decides what happens past the borders: use
    a clamp-to-edge sampler.
use:
    - marbleEncode(uv: vec2f) -> vec4f
    - marbleDecode(encoded: vec4f) -> vec2f
    - marble(ink: texture_2d<f32>, map: texture_2d<f32>, samp: sampler, st: vec2f) -> vec4f
    - marbleReset(frame: f32, period: f32) -> bool
    - marbleBake(frame: f32, period: f32) -> bool
    - marbleReset1(frame: f32) -> bool // period defaults to MARBLE_PERIOD
    - marbleBake1(frame: f32) -> bool  // period defaults to MARBLE_PERIOD
examples:
    - /shaders/splat_strokes_disolve
    - /shaders/splat_strokes_solvent
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// Frames between ink bakes
@if(MARBLE_PERIOD)
const PERIOD: f32 = constants::MARBLE_PERIOD;
@else
const PERIOD: f32 = 60.0;

// Pack a uv in [-0.5, 1.5] into RGBA (hi/lo bytes for x in R,G and y in B,A)
fn marbleEncode(uv: vec2f) -> vec4f {
    let m = clamp((uv + 0.5) * 0.5, vec2f(0.0), vec2f(1.0)) * 255.0;
    let hi = floor(m);
    return vec4f(hi.x / 255.0, m.x - hi.x, hi.y / 255.0, m.y - hi.y);
}

// Inverse of marbleEncode()
fn marbleDecode(c: vec4f) -> vec2f {
    return (c.rb * 255.0 + c.ga) / 255.0 * 2.0 - 0.5;
}

// Read the ink where the map says this pixel came from
fn marble(ink: texture_2d<f32>, map: texture_2d<f32>, samp: sampler, st: vec2f) -> vec4f {
    let uv = marbleDecode(textureSampleLevel(map, samp, st, 0.0));
    return textureSampleLevel(ink, samp, uv, 0.0);
}

// First frame of a cycle (and frame 0): the map restarts from identity
fn marbleReset(frame: f32, period: f32) -> bool {
    return frame < 1.0 || (frame % period) <= 0.5;
}

// Last frame of a cycle (and frame 0): bake the displaced ink into the ink buffer
fn marbleBake(frame: f32, period: f32) -> bool {
    return frame < 1.0 || (frame % period) > period - 1.5;
}

fn marbleReset1(frame: f32) -> bool { return marbleReset(frame, PERIOD); }
fn marbleBake1(frame: f32) -> bool  { return marbleBake(frame, PERIOD); }
