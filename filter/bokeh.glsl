#include "../space/linearizeDepth.glsl"
#include "../color/palette/heatmap.glsl"

/*
author: Dennis Gustafsson
description:  http://blog.tuxedolabs.com/2018/05/04/bokeh-depth-of-field-in-single-pass.html
use: bokeh(<sampler2D> texture, <sampler2D> depth, <vec2> st, <float> focusPoint, <float> focusScale)
options:
    BOKEH_TYPE:
    BOKEH_BLUR_SIZE:
    BOKEH_RAD_SCALE:
    BOKEH_DEPTH_FNC(UV):
    BOKEH_COLOR_FNC(UV):
*/

#ifndef FNC_BOKEH
#define FNC_BOKEH

#ifndef BOKEH_BLUR_SIZE
#define BOKEH_BLUR_SIZE 6.
#endif

// Smaller = nicer blur, larger = faster
#ifndef BOKEH_RAD_SCALE
#define BOKEH_RAD_SCALE .5
#endif

#ifndef GOLDEN_ANGLE
#define GOLDEN_ANGLE 2.39996323
#endif

#ifndef BOKEH_DEPTH_FNC
#define BOKEH_DEPTH_FNC(UV)texture2D(texDepth,UV).r
#endif

#ifndef BOKEH_COLOR_FNC
#define BOKEH_COLOR_FNC(UV)texture2D(tex,UV).rgb
#endif

#ifndef BOKEH_TYPE
#define BOKEH_TYPE vec3
#endif

float getBlurSize(float depth,float focusPoint,float focusScale){
    float coc = clamp((1./focusPoint-1./depth)*focusScale,-1.,1.);
    return abs(coc) * BOKEH_BLUR_SIZE;
}

#ifdef PLATFORM_WEBGL

BOKEH_TYPE bokeh(sampler2D tex,sampler2D texDepth,vec2 texCoord,float focusPoint,float focusScale){
    float pct=0.;
    
    float centerDepth = BOKEH_DEPTH_FNC(texCoord);
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec2 pixelSize = 1.0/u_resolution.xy;
    BOKEH_TYPE color = BOKEH_COLOR_FNC(texCoord);
    
    float total = 1.0;
    float radius = BOKEH_RAD_SCALE;
    for (float angle = 0.0 ; angle < 60.; angle += GOLDEN_ANGLE){
        if (radius >= BOKEH_BLUR_SIZE)
            break;

        vec2 tc = texCoord + vec2(cos(angle), sin(angle)) * pixelSize * radius;
        float sampleDepth = BOKEH_DEPTH_FNC(tc);
        float sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize=clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        BOKEH_TYPE sampleColor = BOKEH_COLOR_FNC(tc);
        #ifdef BOKEH_DEBUG
        sampleColor.rgb = heatmap(pct*0.5+(angle/BOKEH_BLUR_SIZE)*0.1);
        #endif
        color += mix(color/total, sampleColor, pct);
        total += 1.0;
        radius += BOKEH_RAD_SCALE/radius;
    }
    return color/=total;
}

#else

BOKEH_TYPE bokeh(sampler2D tex, sampler2D texDepth, vec2 texCoord, float focusPoint, float focusScale) {
    float pct = 0.0;
    float ang = 0.0;

    float centerDepth = BOKEH_DEPTH_FNC(texCoord);
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec2 pixelSize = 1./u_resolution.xy;
    BOKEH_TYPE color = BOKEH_COLOR_FNC(texCoord);

    float tot = 1.0;
    float radius = BOKEH_RAD_SCALE;
    for (ang = 0.0; radius < BOKEH_BLUR_SIZE; ang += GOLDEN_ANGLE) {
        vec2 tc = texCoord + vec2(cos(ang), sin(ang)) * pixelSize * radius;
        float sampleDepth = BOKEH_DEPTH_FNC(tc);
        float sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize = clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        BOKEH_TYPE sampleColor = BOKEH_COLOR_FNC(tc);
        #ifdef BOKEH_DEBUG
        sampleColor.rgb = heatmap(pct * 0.5 + (ang/BOKEH_BLUR_SIZE) * 0.1);
        #endif
        color += mix(color/tot, sampleColor, pct);
        tot += 1.0;
        radius += BOKEH_RAD_SCALE/radius;
    }
    return color /= tot;
}

#endif

#endif