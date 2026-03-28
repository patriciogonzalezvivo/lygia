/*
contributors: Ricardo Caballero
description: |
    Pack a float into a 4D vector. From https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl
*/

fn pack(v: f32) -> vec4f {
    const PackUpscale: f32 = 256. / 255.; // fraction -> 0..1 (including 1)
    const PackFactors = vec3f( 256. * 256. * 256., 256. * 256.,  256. );
    const ShiftRight8: f32 = 1. / 256.;
	var r = vec4( fract( v * PackFactors ), v );
	r.yzw -= r.xyz * ShiftRight8; // tidy overflow
	return r * PackUpscale;
}
