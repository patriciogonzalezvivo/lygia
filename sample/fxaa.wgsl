#include "../sampler.wgsl"

/*
contributors: [Armin Ronacher, Matt DesLauriers]
description: Basic FXAA implementation based on the code on geeks3d.com with the modification that the texture2DLod stuff was removed since it's unsupported by WebGL from https://github.com/mitsuhiko/webgl-meincraft
use: sampleFXAA(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEFXAA_REDUCE_MIN
    - SAMPLEFXAA_REDUCE_MUL
    - SAMPLEFXAA_SPAN_MAX
    - SAMPLEFXAA_SAMPLE_FNC
license: 
    - BSD licensed (BSD) Copyright (c) 2011 by Armin Ronacher
    - MIT License (MIT) Copyright (c) 2014 Matt DesLauriers
*/

// const SAMPLEFXAA_REDUCE_MIN: f32 = 1.0/128.0;  // TODO: verify

// const SAMPLEFXAA_REDUCE_MUL: f32 = 1.0/8.0;  // TODO: verify

// #define SAMPLEFXAA_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

fn sampleFXAA(tex: SAMPLER_TYPE, uv: vec2f, pixel: vec2f) -> vec4f {
    const SAMPLEFXAA_SPAN_MAX: f32 = 8.0;
    let rgbNW = SAMPLEFXAA_SAMPLE_FNC(tex,uv.xy + vec2f( -1.0, -1.0 ) * pixel).xyz;
    let rgbNE = SAMPLEFXAA_SAMPLE_FNC(tex,uv.xy + vec2f( 1.0, -1.0 ) * pixel).xyz;
    let rgbSW = SAMPLEFXAA_SAMPLE_FNC(tex,uv.xy + vec2f( -1.0, 1.0 ) * pixel).xyz;
    let rgbSE = SAMPLEFXAA_SAMPLE_FNC(tex,uv.xy + vec2f( 1.0, 1.0 ) * pixel).xyz;
    let rgbaM = SAMPLEFXAA_SAMPLE_FNC(tex,uv.xy  * pixel);
    let rgbM = rgbaM.xyz;
    let luma = vec3f( 0.299, 0.587, 0.114 );
    let lumaNW = dot( rgbNW, luma );
    let lumaNE = dot( rgbNE, luma );
    let lumaSW = dot( rgbSW, luma );
    let lumaSE = dot( rgbSE, luma );
    let lumaM = dot( rgbM,  luma );
    let lumaMin = min( lumaM, min( min( lumaNW, lumaNE ), min( lumaSW, lumaSE ) ) );
    let lumaMax = max( lumaM, max( max( lumaNW, lumaNE) , max( lumaSW, lumaSE ) ) );
    vec2 dir = vec2f(-((lumaNW + lumaNE) - (lumaSW + lumaSE)),
                     ((lumaNW + lumaSW) - (lumaNE + lumaSE)) );

    float dirReduce = max(  ( lumaNW + lumaNE + lumaSW + lumaSE ) * ( 0.25 * SAMPLEFXAA_REDUCE_MUL ), 
                            SAMPLEFXAA_REDUCE_MIN );
    let rcpDirMin = 1.0 / ( min( abs( dir.x ), abs( dir.y ) ) + dirReduce );
    dir = min( vec2f(SAMPLEFXAA_SPAN_MAX,  SAMPLEFXAA_SPAN_MAX),
                max(vec2f(-SAMPLEFXAA_SPAN_MAX, -SAMPLEFXAA_SPAN_MAX),
                    dir * rcpDirMin)) * pixel;

    vec4 rgbA = 0.5 * ( SAMPLEFXAA_SAMPLE_FNC(tex, uv.xy + dir * (1.0/3.0 - 0.5)) +
                        SAMPLEFXAA_SAMPLE_FNC(tex, uv.xy + dir * (2.0/3.0 - 0.5)) );
    vec4 rgbB = rgbA * 0.5 + 0.25 * (
                        SAMPLEFXAA_SAMPLE_FNC(tex, uv.xy + dir * -0.5) +
                        SAMPLEFXAA_SAMPLE_FNC(tex, uv.xy + dir * 0.5) );

    let lumaB = dot(rgbB, vec4f(luma, 0.0));
    if ( ( lumaB < lumaMin ) || ( lumaB > lumaMax ) )
        return rgbA;
    else
        return rgbB;
}
