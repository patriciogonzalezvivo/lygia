#include "../sample.glsl"

/*
original_author: Armin Ronacher
description: Basic FXAA implementation based on the code on geeks3d.com with the modification that the texture2DLod stuff was removed since it's unsupported by WebGL from https://github.com/mitsuhiko/webgl-meincraft
use: sampleFXAA(<sampler2D> tex, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEFXAA_REDUCE_MIN
    - SAMPLEFXAA_REDUCE_MUL
    - SAMPLEFXAA_SPAN_MAX
    - SAMPLEFXAA_SAMPLE_FNC
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
vec4 sampleFXAA(sampler2D tex, vec2 uv, vec2 pixel) {
    vec3 rgbNW  = SAMPLEFXAA_SAMPLE_FNC(uv.xy + vec2( -1.0, -1.0 ) * pixel).xyz;
    vec3 rgbNE  = SAMPLEFXAA_SAMPLE_FNC(uv.xy + vec2( 1.0, -1.0 ) * pixel).xyz;
    vec3 rgbSW  = SAMPLEFXAA_SAMPLE_FNC(uv.xy + vec2( -1.0, 1.0 ) * pixel).xyz;
    vec3 rgbSE  = SAMPLEFXAA_SAMPLE_FNC(uv.xy + vec2( 1.0, 1.0 ) * pixel).xyz;
    vec4 rgbaM  = SAMPLEFXAA_SAMPLE_FNC(uv.xy  * pixel);
    vec3 rgbM   = rgbaM.xyz;
    vec3 luma   = vec3( 0.299, 0.587, 0.114 );
    float lumaNW    = dot( rgbNW, luma );
    float lumaNE    = dot( rgbNE, luma );
    float lumaSW    = dot( rgbSW, luma );
    float lumaSE    = dot( rgbSE, luma );
    float lumaM     = dot( rgbM,  luma );
    float lumaMin   = min( lumaM, min( min( lumaNW, lumaNE ), min( lumaSW, lumaSE ) ) );
    float lumaMax   = max( lumaM, max( max( lumaNW, lumaNE) , max( lumaSW, lumaSE ) ) );
    vec2 dir = vec2(-((lumaNW + lumaNE) - (lumaSW + lumaSE)),
                     ((lumaNW + lumaSW) - (lumaNE + lumaSE)) );

    float dirReduce = max(  ( lumaNW + lumaNE + lumaSW + lumaSE ) * ( 0.25 * SAMPLEFXAA_REDUCE_MUL ), 
                            SAMPLEFXAA_REDUCE_MIN );
    float rcpDirMin = 1.0 / ( min( abs( dir.x ), abs( dir.y ) ) + dirReduce );
    dir = min( vec2(SAMPLEFXAA_SPAN_MAX,  SAMPLEFXAA_SPAN_MAX),
                max(vec2(-SAMPLEFXAA_SPAN_MAX, -SAMPLEFXAA_SPAN_MAX),
                    dir * rcpDirMin)) * pixel;

    vec4 rgbA = 0.5 * ( SAMPLEFXAA_SAMPLE_FNC( uv.xy + dir * (1.0/3.0 - 0.5)) +
                        SAMPLEFXAA_SAMPLE_FNC( uv.xy + dir * (2.0/3.0 - 0.5)) );
    vec4 rgbB = rgbA * 0.5 + 0.25 * (
                        SAMPLEFXAA_SAMPLE_FNC( uv.xy + dir * -0.5) +
                        SAMPLEFXAA_SAMPLE_FNC( uv.xy + dir * 0.5) );

    float lumaB = dot(rgbB, vec4(luma, 0.0));
    if ( ( lumaB < lumaMin ) || ( lumaB > lumaMax ) )
        return rgbA;
    else
        return rgbB;
}

#endif
