#include "../color/palette/heatmap.glsl"

/*
author: Dennis Gustafsson
description:  http://blog.tuxedolabs.com/2018/05/04/bokeh-depth-of-field-in-single-pass.html
use: textureDoF(<sampler2D> texture, <sampler2D> depth, <vec2> st, <float> focusPoint, <float> focusScale)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - TEXTUREDOF_TYPE:
    - TEXTUREDOF_BLUR_SIZE:
    - TEXTUREDOF_RAD_SCALE:
    - TEXTUREDOF_DEPTH_FNC(UV):
    - TEXTUREDOF_COLOR_FNC(UV):
    
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef FNC_TEXTUREDOF
#define FNC_TEXTUREDOF

#ifndef TEXTUREDOF_BLUR_SIZE
#define TEXTUREDOF_BLUR_SIZE 6.
#endif

// Smaller = nicer blur, larger = faster
#ifndef TEXTUREDOF_RAD_SCALE
#define TEXTUREDOF_RAD_SCALE .5
#endif

#ifndef GOLDEN_ANGLE
#define GOLDEN_ANGLE 2.39996323
#endif

#ifndef TEXTUREDOF_DEPTH_FNC
#define TEXTUREDOF_DEPTH_FNC(UV) SAMPLER_FNC(texDepth,UV).r
#endif

#ifndef TEXTUREDOF_COLOR_FNC
#define TEXTUREDOF_COLOR_FNC(UV) SAMPLER_FNC(tex,UV).rgb
#endif

#ifndef TEXTUREDOF_TYPE
#define TEXTUREDOF_TYPE vec3
#endif

float getBlurSize(float depth,float focusPoint,float focusScale){
    float coc = clamp((1./focusPoint-1./depth)*focusScale,-1.,1.);
    return abs(coc) * TEXTUREDOF_BLUR_SIZE;
}

#ifdef PLATFORM_WEBGL

TEXTUREDOF_TYPE textureDoF(sampler2D tex,sampler2D texDepth,vec2 texCoord,float focusPoint,float focusScale){
    float pct=0.;
    
    float centerDepth = TEXTUREDOF_DEPTH_FNC(texCoord);
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec2 pixelSize = 1.0/u_resolution.xy;
    TEXTUREDOF_TYPE color = TEXTUREDOF_COLOR_FNC(texCoord);
    
    float total = 1.0;
    float radius = TEXTUREDOF_RAD_SCALE;
    for (float angle = 0.0 ; angle < 60.; angle += GOLDEN_ANGLE){
        if (radius >= TEXTUREDOF_BLUR_SIZE)
            break;

        vec2 tc = texCoord + vec2(cos(angle), sin(angle)) * pixelSize * radius;
        float sampleDepth = TEXTUREDOF_DEPTH_FNC(tc);
        float sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize=clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        TEXTUREDOF_TYPE sampleColor = TEXTUREDOF_COLOR_FNC(tc);
        #ifdef TEXTUREDOF_DEBUG
        sampleColor.rgb = heatmap(pct*0.5+(angle/TEXTUREDOF_BLUR_SIZE)*0.1);
        #endif
        color += mix(color/total, sampleColor, pct);
        total += 1.0;
        radius += TEXTUREDOF_RAD_SCALE/radius;
    }
    return color/=total;
}

#else

TEXTUREDOF_TYPE textureDoF(sampler2D tex, sampler2D texDepth, vec2 texCoord, float focusPoint, float focusScale) {
    float pct = 0.0;
    float ang = 0.0;

    float centerDepth = TEXTUREDOF_DEPTH_FNC(texCoord);
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec2 pixelSize = 1./u_resolution.xy;
    TEXTUREDOF_TYPE color = TEXTUREDOF_COLOR_FNC(texCoord);

    float tot = 1.0;
    float radius = TEXTUREDOF_RAD_SCALE;
    for (ang = 0.0; radius < TEXTUREDOF_BLUR_SIZE; ang += GOLDEN_ANGLE) {
        vec2 tc = texCoord + vec2(cos(ang), sin(ang)) * pixelSize * radius;
        float sampleDepth = TEXTUREDOF_DEPTH_FNC(tc);
        float sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize = clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        TEXTUREDOF_TYPE sampleColor = TEXTUREDOF_COLOR_FNC(tc);
        #ifdef TEXTUREDOF_DEBUG
        sampleColor.rgb = heatmap(pct * 0.5 + (ang/TEXTUREDOF_BLUR_SIZE) * 0.1);
        #endif
        color += mix(color/tot, sampleColor, pct);
        tot += 1.0;
        radius += TEXTUREDOF_RAD_SCALE/radius;
    }
    return color /= tot;
}

#endif

#endif