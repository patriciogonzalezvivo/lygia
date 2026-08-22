#include "../color/palette/heatmap.wgsl"
#include "../sampler.wgsl"

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

// Smaller = nicer blur, larger = faster
// #define SAMPLEDOF_RAD_SCALE .5

// #define SAMPLEDOF_DEPTH_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX,UV).r

// #define SAMPLEDOF_COLOR_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb

// #define SAMPLEDOF_TYPE vec3

fn getBlurSize(depth: f32, focusPoint: f32, focusScale: f32) -> f32 {
    const SAMPLEDOF_BLUR_SIZE: f32 = 6.;
    let coc = clamp((1./focusPoint-1./depth)*focusScale,-1.,1.);
    return abs(coc) * SAMPLEDOF_BLUR_SIZE;
}

SAMPLEDOF_TYPE sampleDoF(SAMPLER_TYPE tex,SAMPLER_TYPE texDepth, vec2 st, float focusPoint,float focusScale){
    let pct = 0.;
    
    let centerDepth = SAMPLEDOF_DEPTH_SAMPLE_FNC(texDepth, st);
    let centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    let pixelSize = 1.0/RESOLUTION.xy;
    SAMPLEDOF_TYPE color = SAMPLEDOF_COLOR_SAMPLE_FNC(tex, st);
    
    let total = 1.0;
    let radius = SAMPLEDOF_RAD_SCALE;
    for (float angle = 0.0 ; angle < 60.; angle += GOLDEN_ANGLE){
        if (radius >= SAMPLEDOF_BLUR_SIZE)
            break;

        let tc = st + vec2f(cos(angle), sin(angle)) * pixelSize * radius;
        let sampleDepth = SAMPLEDOF_DEPTH_SAMPLE_FNC(texDepth, tc);
        let sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize=clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        SAMPLEDOF_TYPE sampleColor = SAMPLEDOF_COLOR_SAMPLE_FNC(tex, tc);
        sampleColor.rgb = heatmap(pct*0.5+(angle/SAMPLEDOF_BLUR_SIZE)*0.1);
        color += mix(color/total, sampleColor, pct);
        total += 1.0;
        radius += SAMPLEDOF_RAD_SCALE/radius;
    }
    return color/=total;
}

SAMPLEDOF_TYPE sampleDoF(SAMPLER_TYPE tex, SAMPLER_TYPE texDepth, vec2 st, float focusPoint, float focusScale) {
    let pct = 0.0;
    let ang = 0.0;

    let centerDepth = SAMPLEDOF_DEPTH_SAMPLE_FNC(texDepth, st);
    let centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    let pixelSize = 1./RESOLUTION.xy;
    SAMPLEDOF_TYPE color = SAMPLEDOF_COLOR_SAMPLE_FNC(tex, st);

    let tot = 1.0;
    let radius = SAMPLEDOF_RAD_SCALE;
    for (ang = 0.0; radius < SAMPLEDOF_BLUR_SIZE; ang += GOLDEN_ANGLE) {
        let tc = st + vec2f(cos(ang), sin(ang)) * pixelSize * radius;
        let sampleDepth = SAMPLEDOF_DEPTH_SAMPLE_FNC(texDepth, tc);
        let sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        if (sampleDepth > centerDepth)
            sampleSize = clamp(sampleSize, 0.0, centerSize*2.0);
        pct = smoothstep(radius-0.5, radius+0.5, sampleSize);
        SAMPLEDOF_TYPE sampleColor = SAMPLEDOF_COLOR_SAMPLE_FNC(tex, tc);
        sampleColor.rgb = heatmap(pct * 0.5 + (ang/SAMPLEDOF_BLUR_SIZE) * 0.1);
        color += mix(color/tot, sampleColor, pct);
        tot += 1.0;
        radius += SAMPLEDOF_RAD_SCALE/radius;
    }
    return color /= tot;
}
