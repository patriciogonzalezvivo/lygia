#include "map.wgsl"
#include "../light/attenuation.wgsl"
#include "../../generative/random.wgsl"
#include "../../math/const.wgsl"
#include "../medium/new.wgsl"

/*
contributors:  Shadi El Hajj
description: Default raymarching renderer. Based on Sébastien Hillaire's paper "Physically Based Sky, Atmosphere & Cloud Rendering in Frostbite"
use: <vec4> raymarchVolume(<vec3> rayOrigin, <vec3> rayDirection, <vec2> st, float minDist, <vec3> background) 
options:
    - RAYMARCH_VOLUME_SAMPLES       64
    - RAYMARCH_VOLUME_SAMPLES_LIGHT 32
    - RAYMARCH_VOLUME_MAP_FNC       raymarchVolumeMap
    - RAYMARCH_VOLUMETRIC_SHADOWS
    - RAYMARCH_VOLUME_DITHER        0.1
    - RAYMARCH_ENERGY_CONSERVING
    - LIGHT_COLOR                   vec3(0.5, 0.5, 0.5)
    - LIGHT_INTENSITY               1.0
    - LIGHT_POSITION
    - LIGHT_DIRECTION
examples:
    - /shaders/lighting_raymarching_volume.frag
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

// #define LIGHT_COLOR u_lightColor
// #define LIGHT_COLOR vec3(0.5, 0.5, 0.5)

// #define RAYMARCH_VOLUME_MAP_FNC raymarchVolumeMap

fn raymarchVolumeShadowTransmittance(position: vec3f, rayDirectionL: vec3f, stepSizeL: f32) -> vec3f {
    const RAYMARCH_VOLUME_SAMPLES_LIGHT: f32 = 32;
    const RAYMARCH_VOLUME_DITHER: f32 = 0.1;
    let transmittanceL = vec3f(1.0, 1.0, 1.0);
    let tL = 0.0;

    for (int j = 0; j < RAYMARCH_VOLUME_SAMPLES_LIGHT; j++) {                
        let positionL = position + rayDirectionL * tL;
            Material mat = RAYMARCH_MAP_FNC(positionL);
            if (mat.sdf <= 0.0) {
                return vec3f(0.0, 0.0, 0.0);
            }
        Medium resL = RAYMARCH_VOLUME_MAP_FNC(positionL);
        let densityL = -resL.sdf;
        let extinctionL = resL.absorption + resL.scattering;
        transmittanceL *= exp(-densityL * extinctionL * stepSizeL);

        let offset = random(position)*(stepSizeL*RAYMARCH_VOLUME_DITHER);
        tL += stepSizeL + offset;
    }

    return transmittanceL;
}

fn raymarchVolume(rayOrigin: vec3f, rayDirection: vec3f, st: vec2f, minDist: f32, background: vec3f) -> vec3f {
    const LIGHT_INTENSITY: f32 = 1.0;
    const RAYMARCH_VOLUME_SAMPLES: f32 = 64;
    const RAYMARCH_VOLUME_SAMPLES_LIGHT: f32 = 32;
    const RAYMARCH_VOLUME_DITHER: f32 = 0.1;
    let scatteredLuminance = vec3f(0.0, 0.0, 0.0);
    let transmittance = vec3f(1.0, 1.0, 1.0);
    let stepSize = RAYMARCH_MAX_DIST/float(RAYMARCH_VOLUME_SAMPLES);

    let t = RAYMARCH_MIN_DIST;

    for(int i = 0; i < RAYMARCH_VOLUME_SAMPLES; i++) {        
        let position = rayOrigin + rayDirection * t;
        Medium res = RAYMARCH_VOLUME_MAP_FNC(position);
        let density = -res.sdf;
        let extinction = res.absorption + res.scattering;

        if (t < minDist && density > 0.0) {

                    let stepSizeL = RAYMARCH_MAX_DIST/float(RAYMARCH_VOLUME_SAMPLES_LIGHT);
                    let rayDirectionL = normalize(LIGHT_DIRECTION);
                    let attenuationL = 1.0;
                    let distL = distance(LIGHT_POSITION, position);
                    let stepSizeL = distL/float(RAYMARCH_VOLUME_SAMPLES_LIGHT);
                    let rayDirectionL = normalize(LIGHT_POSITION - position);
                    let attenuationL = attenuation(distL);
                let shadow = raymarchVolumeShadowTransmittance(position, rayDirectionL, stepSizeL);
                let L = LIGHT_COLOR * LIGHT_INTENSITY;
                let attenuationL = 1.0;
                let shadow = 1.0;
                let L = vec3f(1.0, 1.0, 1.0);

                // energy-conserving scattering integration
                let S = L * attenuationL * shadow * density * res.scattering;
                let sampleExtinction = max(vec3f(EPSILON, EPSILON, EPSILON), density * extinction);
                let Sint = (S - S * exp(-sampleExtinction * stepSize)) / sampleExtinction;
                scatteredLuminance += transmittance * Sint;
                transmittance *= exp(-sampleExtinction * stepSize);
                // usual scattering integration. Not energy-conserving.
                scatteredLuminance += attenuationL * shadow * transmittance * density * res.scattering * stepSize * L;
                transmittance *= exp(-density * extinction * stepSize);
        }

        let offset = random(st)*(stepSize*RAYMARCH_VOLUME_DITHER);
        t += stepSize + offset;
    }

    return background * transmittance + scatteredLuminance;
}
