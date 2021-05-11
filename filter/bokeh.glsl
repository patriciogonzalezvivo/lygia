#include "../space/linearizeDepth.glsl"
#include "../color/palette/heatmap.glsl"

/*
author: Dennis Gustafsson
description:  http://blog.tuxedolabs.com/2018/05/04/bokeh-depth-of-field-in-single-pass.html
use: bokeh(<sampler2D> texture, <sampler2D> depth, <vec2> st, <float> focusPoint, <float> focusScale)
options:
    BOKEH_BLUR_SIZE:
    EDGE_SAMPLER_FNC:
    BOKEH_DEPTH_FNC(UV): 
    BOKEH_COLOR_FNC(UV):
*/

#ifndef FNC_BOKEH
#define FNC_BOKEH

#ifndef BOKEH_BLUR_SIZE
#define BOKEH_BLUR_SIZE 6.0
#endif

// Smaller = nicer blur, larger = faster
#ifndef BOKEH_RAD_SCALE
#define BOKEH_RAD_SCALE 0.5
#endif

#ifndef GOLDEN_ANGLE
#define GOLDEN_ANGLE 2.39996323
#endif

#ifndef BOKEH_DEPTH_FNC
#define BOKEH_DEPTH_FNC(UV) texture2D(texDepth, UV).r
#endif

#ifndef BOKEH_COLOR_FNC
#define BOKEH_COLOR_FNC(UV) texture2D(tex, UV)
#endif

float getBlurSize(float depth, float focusPoint, float focusScale) {
    float coc = clamp((1.0 / focusPoint - 1.0 / depth)*focusScale, -1.0, 1.0);
    return abs(coc) * BOKEH_BLUR_SIZE;
}

#ifdef PLATFORM_WEBGL

vec3 bokeh(sampler2D tex,sampler2D texDepth,vec2 texCoord,float focusPoint,float focusScale){
    float pct=0.;
    
    float centerDepth = BOKEH_DEPTH_FNC(texCoord);
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec2 pixelSize = 1.0/u_resolution.xy;
    vec3 color = BOKEH_COLOR_FNC(texCoord).rgb;
    
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
        vec3 sampleColor = BOKEH_COLOR_FNC(tc).rgb;
        #ifdef BOKEH_DEBUG
        sampleColor = heatmap(pct*0.5+(angle/BOKEH_BLUR_SIZE)*0.1);
        #endif
        color += mix(color/total, sampleColor, pct);
        total += 1.0;
        radius += BOKEH_RAD_SCALE/radius;
    }
    return color/=total;
}

#else

vec3 bokeh(sampler2D tex, sampler2D texDepth, vec2 texCoord, float focusPoint, float focusScale) {
    float pct = 0.0;
    float ang = 0.0;

    float centerDepth = BOKEH_DEPTH_FNC(texCoord);
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec2 pixelSize = 1./u_resolution.xy;
    vec3 color = BOKEH_COLOR_FNC(texCoord).rgb;

    float tot = 1.0;
    float radius = BOKEH_RAD_SCALE;
    for (ang = 0.0; radius < BOKEH_BLUR_SIZE; ang += GOLDEN_ANGLE) {
        vec2 tc = texCoord + vec2(cos(ang), sin(ang)) * pixelSize * radius;
        float sampleDepth = BOKEH_DEPTH_FNC(tc);
        float sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize = clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        vec3 sampleColor = BOKEH_COLOR_FNC(tc).rgb;
        #ifdef BOKEH_DEBUG
        sampleColor = heatmap(pct * 0.5 + (ang/BOKEH_BLUR_SIZE) * 0.1);
        #endif
        color += mix(color/tot, sampleColor, pct);
        tot += 1.0;
        radius += BOKEH_RAD_SCALE/radius;
    }
    return color /= tot;
}

#endif

#endif