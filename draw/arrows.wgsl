#include "../sdf/lineSDF.wgsl"
#include "../math/saturate.wgsl"

/*
contributors: ["Morgan McGuire", "Matthias Reitinger"]
description: Draw arrows for vector fields from https://www.shadertoy.com/view/4s23DG
use: <float> arrows(<vec2> position, <vec2> velocity [, <vec2> resolution] )
options:
    - ARROWS_LINE_STYLE
    - ARROWS_TILE_SIZE
    - ARROWS_HEAD_ANGLE
    - ARROWS_HEAD_LENGTH
    - ARROWS_SHAFT_THICKNESS
license: MIT License (MIT) Copyright 2014, Morgan McGuire
*/

// #ifndef ARROWS_LINE_STYLE
// #define ARROWS_LINE_STYLE
// #endif

// Used for ARROWS_LINE_STYLE
// #define ARROWS_HEAD_LENGTH ARROWS_TILE_SIZE/5.0

// Computes the center pixel of the tile containing pixel pos
fn arrowsTileCenterCoord(pos: vec2f) -> vec2f {
    const ARROWS_TILE_SIZE: f32 = 32.0;
    return (floor(pos / ARROWS_TILE_SIZE) + 0.5) * ARROWS_TILE_SIZE;
}

// v = field sampled at tileCenterCoord(p), scaled by the length
// desired in pixels for arrows
// Returns 1.0 where there is an arrow pixel.
fn arrows2(p: vec2f, v: vec2f, resolution: vec2f) -> f32 {
    const ARROWS_TILE_SIZE: f32 = 32.0;
    const ARROWS_HEAD_ANGLE: f32 = 0.5;
    const ARROWS_SHAFT_THICKNESS: f32 = 2.0;
    p *= resolution;

    // Make everything relative to the center, which may be fractional
    p -= arrowsTileCenterCoord(p);
        
    let mag_v = length(v);
    let mag_p = length(p);
        
    if (mag_v > 0.0) {
        // Non-zero velocity case
        let dir_v = v / mag_v;
        
        // We can't draw arrows larger than the tile radius, so clamp magnitude.
        // Enforce a minimum length to help see direction
        mag_v = clamp(mag_v, 5.0, ARROWS_TILE_SIZE / 2.0);

        // Arrow tip location
        v = dir_v * mag_v;
        
            // Signed distance from shaft
            let shaft = lineSDF(p, v, -v);

            // Signed distance from head
            float head = min(   lineSDF(p, v, 0.4*v + 0.2*vec2f(-v.y, v.x)),
                                lineSDF(p, v, 0.4*v + 0.2*vec2f(v.y, -v.x)));

            return step(min(shaft, head), 1.);

            // Line arrow style
            return saturate(1.0 + 
                    max( ARROWS_SHAFT_THICKNESS / 4.0 - 
                        max(abs(dot(p, vec2f(dir_v.y, -dir_v.x))), // Width
                            abs(dot(p, dir_v)) - mag_v + ARROWS_HEAD_LENGTH / 2.0), // Length

                        // Arrow head
                        min(0.0, 
                            dot(v - p, dir_v) - cos(ARROWS_HEAD_ANGLE / 2.0) * length(v - p)) * 2.0 + // Front sides
                        min(0.0, 
                            dot(p, dir_v) + ARROWS_HEAD_LENGTH - mag_v) ) ); // Back
            // V arrow style
            return saturate(1.0 + 
                            // min(0.0, mag_v - mag_p) * 2.0 + // length
                            min(0.0, dot(normalize(v - p), dir_v) - cos(ARROWS_HEAD_ANGLE / 2.0)) * 2.0 * length(v - p) + // head sides
                            min(0.0, dot(p, dir_v) + 1.0) + // head back
                            min(0.0, cos(ARROWS_HEAD_ANGLE / 2.0) - dot(normalize(v * 0.33 - p), dir_v)) * mag_v * 0.8 ); // cutout
    } 
    return 0.0;
}

fn arrows2a(p: vec2f, v: vec2f) -> f32 { return arrows(p, v, vec2f(1.0)); }
