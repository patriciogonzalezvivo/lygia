#include "../math/saturate.glsl"
#include "../space/linearizeDepth.glsl"
#include "../space/depth2viewZ.glsl"
#include "../sampler.glsl"
#include "../generative/random.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: ScreenSpace Ambient Occlusion
use: <float> ssao(<SAMPLER_TYPE> texPosition, <SAMPLER_TYPE> texNormal, vec2 <st> [, <float> radius, float <bias>])
options:
    - SSAO_SAMPLES_NUM: number of half-sphere offsets samples
    - SSAO_SAMPLES_ARRAY: array of weighted vec3 half-sphere offsets
    - SSAO_NOISE_NUM: number of vec3 noise offsets
    - SSAO_NOISE_ARRAY: array of vec3 noise offsets
    - SSAO_NOISE2_FNC(ST): (random2(ST * 100.) * 0.1)
    - SSAO_NOISE3_FNC(POS): (random3(POS) * 0.3)
    - CAMERA_PROJECTION_MATRIX: camera projection mat4 matrix
    - CAMERA_NEAR_CLIP: required for depth only SSAO
    - CAMERA_FAR_CLIP: required for depth only SSAO
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SSAO_SAMPLES
#define SSAO_SAMPLES 60.0
#endif

#ifndef SSAO_SAMPLES_NUM
#define SSAO_SAMPLES_NUM 8
#endif

#ifndef SSAO_SAMPLES_ARRAY
#if defined(GLSLVIEWER)
#define SSAO_SAMPLES_ARRAY u_ssaoSamples
uniform vec3 u_ssaoSamples[SSAO_SAMPLES_NUM];
#endif
#endif

#ifndef SSAO_NOISE_NUM
#define SSAO_NOISE_NUM 4
#endif

#ifndef SSAO_NOISE_ARRAY
#if defined(GLSLVIEWER)
#define SSAO_NOISE_ARRAY u_ssaoNoise
uniform vec3 u_ssaoNoise[SSAO_NOISE_NUM];
#endif
#endif

#ifndef SSAO_NOISE2_FNC
#define SSAO_NOISE2_FNC(ST) random2(ST)
#endif

#ifndef SSAO_NOISE3_FNC
#define SSAO_NOISE3_FNC(POS) random3(POS)
#endif

#ifndef SSAO_DEPTH_BIAS
#define SSAO_DEPTH_BIAS 0.05
#endif

#ifndef SSAO_DEPTH_SAMPLE_FNC
#define SSAO_DEPTH_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r
#endif

#ifndef SSAO_POS_SAMPLE_FNC
#define SSAO_POS_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef SSAO_NORMAL_SAMPLE_FNC
#define SSAO_NORMAL_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).xyz
#endif

#ifndef FNC_SSAO
#define FNC_SSAO

#if defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)

float ssao(SAMPLER_TYPE texDepth, vec2 st, vec2 pixel, float radius) {

    #if defined(SSAO_NOISE_ARRAY)
    float noiseS    = sqrt(float(SSAO_NOISE_NUM));
    int  noiseX     = int( mod(gl_FragCoord.x - 0.5, noiseS) );
    int  noiseY     = int( mod(gl_FragCoord.y - 0.5, noiseS) );
    vec2 noise      = SSAO_NOISE_ARRAY[noiseX + noiseY * int(noiseS)].xy;
    #else
    vec2 noise = SSAO_NOISE2_FNC( st ); 
    #endif
    noise *= 0.1;

    float depth     = depth2viewZ( SSAO_DEPTH_SAMPLE_FNC( texDepth, st ) ) + SSAO_DEPTH_BIAS * 0.5; 
    float ao        = 0.0;

    // if (depth < 0.99) 
    {
        float w = pixel.x / depth + noise.x;
        float h = pixel.y / depth + noise.y;

        float dz = 1.0 / float( SSAO_SAMPLES_NUM ); 
        float l = 0.0; 
        float z = 1.0 - dz * 0.5; 

        for ( int i = 0; i < SSAO_SAMPLES_NUM; i ++ ) { 
            float r = sqrt( 1.0 - z ); 
            float pw = cos( l ) * r; 
            float ph = sin( l ) * r; 
            vec2 vv = radius * vec2( pw * w, ph * h);
            ao += ( step( depth2viewZ( SSAO_DEPTH_SAMPLE_FNC( texDepth, st + vv) ), depth) + 
                    step( depth2viewZ( SSAO_DEPTH_SAMPLE_FNC( texDepth, st - vv) ), depth) ) * 0.5;
            z = z - dz; 
            l = l + 2.399963229728653; 
        } 
        ao = 1.0 - ao * dz;
        ao = saturate( 1.98 * ( 1.0 - ao ) );
    }
    return ao;
}
#endif


float ssao(sampler2D texDepth, vec2 coord) {
    float cd = SSAO_DEPTH_SAMPLE_FNC(texDepth, coord);
    float ao_radious = 1.0/100.0;
    float screenRadius = 0.5 * (ao_radious / cd) / 0.53135;
    float li = 0.0;
    float count = 0.0;
    for(float i=0.0; i< SSAO_SAMPLES; i++) {
        vec3 p = normalize(SSAO_NOISE3_FNC(vec3(coord, i)) * 2.0 - 1.0);
        vec2 sp = vec2(coord.x + p.x * screenRadius, coord.y + p.y * screenRadius);
        float d = SSAO_DEPTH_SAMPLE_FNC(texDepth, sp);
        float at = pow(length(p)-1.0, 2.0);
        li += step(cd + p.z * ao_radious, d) * at;
        count += at;
    }
    return 1.0 - li / count;
}

#if defined(SSAO_SAMPLES_ARRAY) && defined(CAMERA_PROJECTION_MATRIX)

float ssao(SAMPLER_TYPE texPosition, SAMPLER_TYPE texNormal, vec2 st, float radius) {
    vec4  position  = SSAO_POS_SAMPLE_FNC(texPosition, st);
    vec3  normal    = SSAO_NORMAL_SAMPLE_FNC(texNormal, st);

    #if defined(SSAO_NOISE_ARRAY) 
    float noiseS    = sqrt(float(SSAO_NOISE_NUM));
    int   noiseX    = int( mod(gl_FragCoord.x - 0.5, noiseS) );
    int   noiseY    = int( mod(gl_FragCoord.y - 0.5, noiseS) );
    vec3  noise     = SSAO_NOISE_ARRAY[noiseX + noiseY * int(noiseS)];
    #else
    vec3  noise     = SSAO_NOISE3_FNC( position.xyz ); 
    #endif

    vec3 tangent    = normalize(noise - normal * dot(noise, normal));
    vec3 binormal   = cross(normal, tangent);
    mat3 tbn        = mat3(tangent, binormal, normal);

    float occlusion = 0.0;
    for (int i = 0; i < SSAO_SAMPLES_NUM; ++i) {
        vec3 samplePosition = tbn * SSAO_SAMPLES_ARRAY[i];
        samplePosition = position.xyz + samplePosition * radius;

        vec4 offsetUV = vec4(samplePosition, 1.0);
        offsetUV = CAMERA_PROJECTION_MATRIX * offsetUV;
        offsetUV.xy /= offsetUV.w;
        offsetUV.xy = offsetUV.xy * 0.5 + 0.5;

        float sampleDepth = SAMPLER_FNC(texPosition, offsetUV.xy).z;
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(position.z - sampleDepth));
        occlusion += (sampleDepth >= samplePosition.z + SSAO_DEPTH_BIAS ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion /= float(SSAO_SAMPLES_NUM);
    return 1.0-occlusion;
}
#endif

#endif