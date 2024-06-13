#include "../color/palette/heatmap.glsl"
#include "../sampler.glsl"

/*
contributors: Dennis Gustafsson
description:  http://blog.tuxedolabs.com/2018/05/04/bokeh-depth-of-field-in-single-pass.html
use: sampleDoF(<SAMPLER_TYPE> texture, <SAMPLER_TYPE> depth, <vec2> st, <float> focusPoint, <float> focusScale)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEDOF_TYPE:
    - SAMPLEDOF_BLUR_SIZE:
    - SAMPLEDOF_RAD_SCALE:
    - SAMPLEDOF_DEPTH_SAMPLE_FNC(TEX, UV):
    - SAMPLEDOF_COLOR_SAMPLE_FNC(TEX, UV):
    - SAMPLEDOF_DEBUG
    - RESOLUTION
*/

#ifndef FNC_SAMPLEDOF
#define FNC_SAMPLEDOF

#ifndef SAMPLEDOF_BLUR_SIZE
#define SAMPLEDOF_BLUR_SIZE 6.
#endif

// Smaller = nicer blur, larger = faster
#ifndef SAMPLEDOF_RAD_SCALE
#define SAMPLEDOF_RAD_SCALE .5
#endif

#ifndef GOLDEN_ANGLE
#define GOLDEN_ANGLE 2.39996323
#endif

#ifndef SAMPLEDOF_DEPTH_SAMPLE_FNC
#define SAMPLEDOF_DEPTH_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX,UV).r
#endif

#ifndef SAMPLEDOF_COLOR_SAMPLE_FNC
#define SAMPLEDOF_COLOR_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb
#endif

#ifndef SAMPLEDOF_TYPE
#define SAMPLEDOF_TYPE vec3
#endif

float getBlurSize(float depth,float focusPoint,float focusScale){
    float coc = clamp((1./focusPoint-1./depth)*focusScale,-1.,1.);
    return abs(coc) * SAMPLEDOF_BLUR_SIZE;
}

#ifdef PLATFORM_WEBGL

SAMPLEDOF_TYPE sampleDoF(SAMPLER_TYPE tex,SAMPLER_TYPE texDepth, vec2 st, float focusPoint,float focusScale){
    float pct=0.;
    
    float centerDepth = SAMPLEDOF_DEPTH_SAMPLE_FNC(texDepth, st);
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec2 pixelSize = 1.0/RESOLUTION.xy;
    SAMPLEDOF_TYPE color = SAMPLEDOF_COLOR_SAMPLE_FNC(tex, st);
    
    float total = 1.0;
    float radius = SAMPLEDOF_RAD_SCALE;
    for (float angle = 0.0 ; angle < 60.; angle += GOLDEN_ANGLE){
        if (radius >= SAMPLEDOF_BLUR_SIZE)
            break;

        vec2 tc = st + vec2(cos(angle), sin(angle)) * pixelSize * radius;
        float sampleDepth = SAMPLEDOF_DEPTH_SAMPLE_FNC(texDepth, tc);
        float sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize=clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        SAMPLEDOF_TYPE sampleColor = SAMPLEDOF_COLOR_SAMPLE_FNC(tex, tc);
        #ifdef SAMPLEDOF_DEBUG
        sampleColor.rgb = heatmap(pct*0.5+(angle/SAMPLEDOF_BLUR_SIZE)*0.1);
        #endif
        color += mix(color/total, sampleColor, pct);
        total += 1.0;
        radius += SAMPLEDOF_RAD_SCALE/radius;
    }
    return color/=total;
}

#else

SAMPLEDOF_TYPE sampleDoF(SAMPLER_TYPE tex, SAMPLER_TYPE texDepth, vec2 st, float focusPoint, float focusScale) {
    float pct = 0.0;
    float ang = 0.0;

    float centerDepth = SAMPLEDOF_DEPTH_SAMPLE_FNC(texDepth, st);
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec2 pixelSize = 1./RESOLUTION.xy;
    SAMPLEDOF_TYPE color = SAMPLEDOF_COLOR_SAMPLE_FNC(tex, st);

    float tot = 1.0;
    float radius = SAMPLEDOF_RAD_SCALE;
    for (ang = 0.0; radius < SAMPLEDOF_BLUR_SIZE; ang += GOLDEN_ANGLE) {
        vec2 tc = st + vec2(cos(ang), sin(ang)) * pixelSize * radius;
        float sampleDepth = SAMPLEDOF_DEPTH_SAMPLE_FNC(texDepth, tc);
        float sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize = clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        SAMPLEDOF_TYPE sampleColor = SAMPLEDOF_COLOR_SAMPLE_FNC(tex, tc);
        #ifdef SAMPLEDOF_DEBUG
        sampleColor.rgb = heatmap(pct * 0.5 + (ang/SAMPLEDOF_BLUR_SIZE) * 0.1);
        #endif
        color += mix(color/tot, sampleColor, pct);
        tot += 1.0;
        radius += SAMPLEDOF_RAD_SCALE/radius;
    }
    return color /= tot;
}

#endif

#endif