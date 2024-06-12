#include "../sampler.hlsl"

/*
contributors: Armin Ronacher
description: Basic FXAA implementation based on the code on geeks3d.com with the modification that the texture2DLod stuff was removed since it's unsupported by WebGL from https://github.com/mitsuhiko/webgl-meincraft
use: sampleFXAA(<SAMPLER_TYPE> tex, <float2> st, <float2> pixel)
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

#ifndef SAMPLEFXAA_REDUCE_MIN
#define SAMPLEFXAA_REDUCE_MIN   (1.0/128.0)
#endif

#ifndef SAMPLEFXAA_REDUCE_MUL
#define SAMPLEFXAA_REDUCE_MUL   (1.0/8.0)
#endif

#ifndef SAMPLEFXAA_SPAN_MAX
#define SAMPLEFXAA_SPAN_MAX     8.0
#endif

#ifndef SAMPLEFXAA_SAMPLE_FNC
#define SAMPLEFXAA_SAMPLE_FNC(UV) SAMPLER_FNC(tex, UV)
#endif

#ifndef FNC_SAMPLEFXAA
#define FNC_SAMPLEFXAA 
float4 sampleFXAA(SAMPLER_TYPE tex, float2 uv, float2 pixel) {
    float3 rgbNW  = SAMPLEFXAA_SAMPLE_FNC(uv.xy + float2( -1.0, -1.0 ) * pixel).xyz;
    float3 rgbNE  = SAMPLEFXAA_SAMPLE_FNC(uv.xy + float2( 1.0, -1.0 ) * pixel).xyz;
    float3 rgbSW  = SAMPLEFXAA_SAMPLE_FNC(uv.xy + float2( -1.0, 1.0 ) * pixel).xyz;
    float3 rgbSE  = SAMPLEFXAA_SAMPLE_FNC(uv.xy + float2( 1.0, 1.0 ) * pixel).xyz;
    float4 rgbaM  = SAMPLEFXAA_SAMPLE_FNC(uv.xy  * pixel);
    float3 rgbM   = rgbaM.xyz;
    float3 luma   = float3( 0.299, 0.587, 0.114 );
    float lumaNW    = dot( rgbNW, luma );
    float lumaNE    = dot( rgbNE, luma );
    float lumaSW    = dot( rgbSW, luma );
    float lumaSE    = dot( rgbSE, luma );
    float lumaM     = dot( rgbM,  luma );
    float lumaMin   = min( lumaM, min( min( lumaNW, lumaNE ), min( lumaSW, lumaSE ) ) );
    float lumaMax   = max( lumaM, max( max( lumaNW, lumaNE) , max( lumaSW, lumaSE ) ) );
    float2 dir = float2(-((lumaNW + lumaNE) - (lumaSW + lumaSE)),
                     ((lumaNW + lumaSW) - (lumaNE + lumaSE)) );

    float dirReduce = max(  ( lumaNW + lumaNE + lumaSW + lumaSE ) * ( 0.25 * SAMPLEFXAA_REDUCE_MUL ), 
                            SAMPLEFXAA_REDUCE_MIN );
    float rcpDirMin = 1.0 / ( min( abs( dir.x ), abs( dir.y ) ) + dirReduce );
    dir = min( float2(SAMPLEFXAA_SPAN_MAX,  SAMPLEFXAA_SPAN_MAX),
                max(float2(-SAMPLEFXAA_SPAN_MAX, -SAMPLEFXAA_SPAN_MAX),
                    dir * rcpDirMin)) * pixel;

    float4 rgbA = 0.5 * (   SAMPLEFXAA_SAMPLE_FNC( uv.xy + dir * (1.0/3.0 - 0.5)) +
                            SAMPLEFXAA_SAMPLE_FNC( uv.xy + dir * (2.0/3.0 - 0.5)) );
    float4 rgbB = rgbA * 0.5 + 0.25 * ( SAMPLEFXAA_SAMPLE_FNC( uv.xy + dir * -0.5) +
                                        SAMPLEFXAA_SAMPLE_FNC( uv.xy + dir * 0.5) );

    float lumaB = dot(rgbB, float4(luma, 0.0));
    if ( ( lumaB < lumaMin ) || ( lumaB > lumaMax ) )
        return rgbA;
    else
        return rgbB;
}

#endif
