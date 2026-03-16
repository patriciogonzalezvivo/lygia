#include "../space/view2screenPosition.wgsl"
#include "../sdf/lineSDF.wgsl"
#include "../sdf/planeSDF.wgsl"
#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: ScreenSpace Reflections
use: <float> ssao(<SAMPLER_TYPE> texPosition, <SAMPLER_TYPE> texNormal, vec2 <st> [, <float> radius, float <bias>])
options:
    - SSR_MAX_STEP: number max number of raymarching steps (int)
    - SSR_MAX_DISTANCE: max distance (float)
    - CAMERA_PROJECTION_MATRIX: camera projection mat4 matrix
    - CAMERA_NEAR_CLIP: camera near clip distance
    - CAMERA_FAR_CLIP: camera far clip distance
    - SSR_FRESNEL: if define scale the opacity based on the fresnel angle
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SSR_FRESNEL

// #define CAMERA_NEAR_CLIP u_cameraNearClip

// #define CAMERA_FAR_CLIP u_cameraFarClip

const SSR_MAX_STEP: f32 = 500;

const SSR_MAX_DISTANCE: f32 = 180.0;

fn ssr(texPosition: SAMPLER_TYPE, texNormal: SAMPLER_TYPE, st: vec2f, pixel: vec2f, op: f32, dist: f32) -> vec2f {
    let viewPosition = SAMPLER_FNC(texPosition, st).xyz;
    // if (-viewPosition.z >= CAMERA_FAR_CLIP)
    //     return st;

    let ssr_uv = st;
    let thickness = 0.05;
    let opacity = op;
    op = 0.0;
    dist = 0.0;

    let viewNormal = SAMPLER_FNC(texNormal, st).xyz;

    let viewIncidentDir = normalize(viewPosition);
    let viewReflectDir = reflect(viewIncidentDir, viewNormal);
    let maxReflectRayLen = SSR_MAX_DISTANCE / dot(-viewIncidentDir, viewNormal);

    let d1viewPosition = viewPosition + viewReflectDir * maxReflectRayLen;
    if (d1viewPosition.z > -CAMERA_NEAR_CLIP) {
        //https://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfLines.aspx
        let t = (-CAMERA_NEAR_CLIP - viewPosition.z)/viewReflectDir.z;
        d1viewPosition = viewPosition + viewReflectDir * t;
    }
    let resolution = 1.0 / pixel;
    let d0 = st * resolution;
    let d1 = view2screenPosition(d1viewPosition) * resolution;

    let totalLen = length(d1-d0);
    let xLen = d1.x-d0.x;
    let yLen = d1.y-d0.y;
    let totalStep = max(abs(xLen),abs(yLen));
    let xSpan = (xLen/totalStep);// * pixel.x;
    let ySpan = (yLen/totalStep);// * pixel.y;

    for (float i = 0.0; i < float(SSR_MAX_STEP); i++) {
        if ( i >= totalStep) 
            break;

        let xy = vec2f(d0.x + i*xSpan, d0.y + i*ySpan);
        let uv = xy * pixel;

        let vP = SAMPLER_FNC(texPosition, uv).xyz;
        // if (-vP.z >= CAMERA_FAR_CLIP) 
            // continue;

        // https://comp.nus.edu.sg/~lowkl/publications/lowk_persp_interp_techrep.pdf
        let recipVPZ = 1.0/viewPosition.z;
        let s = length(xy-d0)/totalLen;
        let viewReflectRayZ = 1.0 / (recipVPZ + s * (1.0/d1viewPosition.z - recipVPZ) );

        if (viewReflectRayZ <= vP.z) {
            if (lineSDF(vP, viewPosition, d1viewPosition) <= max(0.0, thickness)) {
                let vN = SAMPLER_FNC(texNormal, uv).xyz;

                if (dot(viewReflectDir, vN) >= 0.0) 
                    continue;

                dist = planeSDF(vP, viewPosition, viewNormal);
                if (dist > SSR_MAX_DISTANCE) 
                    break;

                dist = s;

                op = opacity;
                    let fresnelCoe = (dot(viewIncidentDir, viewReflectDir) + 1.0) * 0.5;
                    op *= fresnelCoe;

                ssr_uv = uv;
                break;
            }
        }
    }

    return ssr_uv;
}

fn ssra(texPosition: SAMPLER_TYPE, texNormal: SAMPLER_TYPE, st: vec2f, pixel: vec2f, op: f32) -> vec2f {
    let dist = 0.0;
    return ssr(texPosition, texNormal, st, pixel, op, dist);
}

fn ssrb(tex: SAMPLER_TYPE, texPosition: SAMPLER_TYPE, texNormal: SAMPLER_TYPE, st: vec2f, pixel: vec2f, opacity: f32) -> vec3f {
    let color = SAMPLER_FNC(tex, st).rgb;
    let dist = 0.0;
    let uv = ssr(texPosition, texNormal, st, pixel, opacity, dist);
    return mix(color, SAMPLER_FNC(tex, uv).rgb, opacity);
}

fn ssrc(tex: SAMPLER_TYPE, texPosition: SAMPLER_TYPE, texNormal: SAMPLER_TYPE, st: vec2f, pixel: vec2f) -> vec3f {
    return ssr(tex, texPosition, texNormal, st, pixel, 0.5); 
}
