/*
contributors: Ricardo Caballero
description: |
    Pack a float into a 4D vector. From https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl
*/

const PackUpscale: f32 = 256.0 / 255.0; // fraction -> 0..1 (including 1)
const UnpackDownscale: f32 = 255.0 / 256.0; // 0..1 -> fraction (excluding 1)
const PackFactors = vec3f( 256.0 * 256.0 * 256.0, 256.0 * 256.0,  256.0 );
const UnpackFactors = UnpackDownscale / vec4f( PackFactors, 1.0 );
const ShiftRight8: f32 = 1.0 / 256.0;

fn pack(v: f32) -> vec4f {
    var r = vec4f( fract( v * PackFactors ), v );
    r.y -= r.x * ShiftRight8;
    r.z -= r.y * ShiftRight8;
    r.w -= r.z * ShiftRight8;
    return r * PackUpscale;
}
