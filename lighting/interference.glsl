#include "envMap.glsl"
#include "fresnel.glsl"

/*
original_author: Cornus Ammonis
description: | 
    Thin-Film interferencence is a natural phenomenon that occurs when light “bounces” inside a medium which thickness 
    is comparable to the light wavelength. When this happens, there is a chance that some of the light immediately 
    reflected off the surface will end up interfering with the light refracted inside the medium.
    Based on https://www.shadertoy.com/view/ld3SDl, https://www.shadertoy.com/view/ldGfRz
use: <vec3> interference(<vec3> ray, <vec3> normal, <float> thickness)
options:
    - INTERFERENCE_DISPERSION: dispersion amount
    - INTERFERENCE_IOR: base IOR value specified as a ratio
    - INTERFERENCE_THICKNESS_SCALE: film thickness scaling factor
    - INTERFERENCE_SCALE: thin-film interference scaling factor
    - INTERFERENCE_GAMMA_SCALE: thin-film interference gamma scaling factor
    - INTERFERENCE_FRESNEL_RATIO: fresnel weight for thin-film interference
    - INTERFERENCE_GAMMA_CURVE: default 50.0
    - INTERFERENCE_GREEN_WEIGHT: default 2.8
*/

// dispersion amount
#ifndef INTERFERENCE_DISPERSION
#define INTERFERENCE_DISPERSION 0.05
#endif

// base IOR value specified as a ratio
#ifndef INTERFERENCE_IOR
#define INTERFERENCE_IOR 0.99
#endif

// film thickness scaling factor
#ifndef INTERFERENCE_THICKNESS_SCALE
#define INTERFERENCE_THICKNESS_SCALE 32.0
#endif 

// reflectance scaling factor
#ifndef INTERFERENCE_SCALE
#define INTERFERENCE_SCALE 3.0
#endif

// reflectance gamma scaling factor
#ifndef INTERFERENCE_GAMMA_SCALE
#define INTERFERENCE_GAMMA_SCALE 2.0
#endif

// fresnel weight for reflectance
#ifndef INTERFERENCE_FRESNEL_RATIO
#define INTERFERENCE_FRESNEL_RATIO 0.7
#endif

#ifndef INTERFERENCE_GAMMA_CURVE
#define INTERFERENCE_GAMMA_CURVE 1.0
#endif

#ifndef INTERFERENCE_GREEN_WEIGHT
#define INTERFERENCE_GREEN_WEIGHT 2.8
#endif

#ifndef INTERFERENCE_ENVMAP_FNC
#define INTERFERENCE_ENVMAP_FNC(N, R, M) envMap(N, R, M)
#endif

#ifndef INTERFERENCE_WAVELENGTHS0
#define INTERFERENCE_WAVELENGTHS0 vec3(1.0, 0.8, 0.6)
#endif

#ifndef INTERFERENCE_WAVELENGTHS1
#define INTERFERENCE_WAVELENGTHS1 vec3(0.4, 0.2, 0.0)
#endif

#ifndef FNC_THINFILMINTERFER
#define FNC_THINFILMINTERFER

// Filmic
vec3 interference_gamma(vec3 x) { return log(INTERFERENCE_GAMMA_CURVE * x + 1.0) / INTERFERENCE_GAMMA_SCALE;     }
vec3 interference_gamma_inverse(vec3 y) { return (1.0 / INTERFERENCE_GAMMA_CURVE) * (exp(INTERFERENCE_GAMMA_SCALE * y) - 1.0);  }

vec3 interference_fresnel( vec3 rd, vec3 norm, vec3 n2 ) {
    vec3 f0 = pow((1.0-n2)/(1.0+n2), vec3(2.0));
    return schlick(f0, vec3(1.0), 1.0-saturate(1.0+dot(rd, norm)));
}

// sample weights for the cubemap given a wavelength i
vec3 interference_sampleWeights(float i) { 
    return vec3((1.0 - i) * (1.0 - i), INTERFERENCE_GREEN_WEIGHT * i * (1.0 - i), i * i); 
}
vec3 interference_texCubeSampleWeights(float i) {
    vec3 w = interference_sampleWeights(i);
    return w / dot(w, vec3(1.0));
}

vec3 interference_sampleCubeMap(vec3 i, vec3 rd, float roughness, float metallic) {
    vec3 col = INTERFERENCE_ENVMAP_FNC(rd, roughness, metallic);
    return vec3(
        dot(interference_texCubeSampleWeights(i.x), col),
        dot(interference_texCubeSampleWeights(i.y), col),
        dot(interference_texCubeSampleWeights(i.z), col)
    );
}

vec3 interference_sampleCubeMap(vec3 i, vec3 rd0, vec3 rd1, vec3 rd2, float roughness, float metallic) {
    vec3 col0 = INTERFERENCE_ENVMAP_FNC(rd0, roughness, metallic);
    vec3 col1 = INTERFERENCE_ENVMAP_FNC(rd1, roughness, metallic); 
    vec3 col2 = INTERFERENCE_ENVMAP_FNC(rd2, roughness, metallic);
    return vec3(
        dot(interference_texCubeSampleWeights(i.x), col0),
        dot(interference_texCubeSampleWeights(i.y), col1),
        dot(interference_texCubeSampleWeights(i.z), col2)
    );
}

vec3 interference_attenuation(vec3 wavelengths, vec3 normal, vec3 rd, float thickness) {
    return 0.5 + 0.5 * cos(((INTERFERENCE_THICKNESS_SCALE * thickness)/(wavelengths + 1.0)) * dot(normal, rd));    
}

vec3 interference(vec3 ray, vec3 normal, float thickness, float roughness, float metallic) {
    const vec3 wavelengths0 = INTERFERENCE_WAVELENGTHS0;
    const vec3 wavelengths1 = INTERFERENCE_WAVELENGTHS1;
    const vec3 iors0 = INTERFERENCE_IOR + wavelengths0 * INTERFERENCE_DISPERSION;
    const vec3 iors1 = INTERFERENCE_IOR + wavelengths1 * INTERFERENCE_DISPERSION;

    vec3 att0 = interference_attenuation(wavelengths0, normal, ray, thickness);
    vec3 att1 = interference_attenuation(wavelengths1, normal, ray, thickness);

    vec3 rray = reflect(ray, normal);
    vec3 cube0 = INTERFERENCE_GAMMA_SCALE * att0 * interference_sampleCubeMap(wavelengths0, rray, roughness, metallic);
    vec3 cube1 = INTERFERENCE_GAMMA_SCALE * att1 * interference_sampleCubeMap(wavelengths1, rray, roughness, metallic);

    vec3 f0 = (1.0 - INTERFERENCE_FRESNEL_RATIO) + INTERFERENCE_FRESNEL_RATIO * interference_fresnel(ray, normal, 1.0 / iors0);
    vec3 f1 = (1.0 - INTERFERENCE_FRESNEL_RATIO) + INTERFERENCE_FRESNEL_RATIO * interference_fresnel(ray, normal, 1.0 / iors1);
    vec3 i0 = INTERFERENCE_SCALE * interference_gamma_inverse(cube0 * f0);
    vec3 i1 = INTERFERENCE_SCALE * interference_gamma_inverse(cube1 * f1);

    vec3 w0 = interference_sampleWeights(wavelengths0.x);
    vec3 w1 = interference_sampleWeights(wavelengths0.y);
    vec3 w2 = interference_sampleWeights(wavelengths0.z);
    vec3 w3 = interference_sampleWeights(wavelengths1.x);
    vec3 w4 = interference_sampleWeights(wavelengths1.y);
    vec3 w5 = interference_sampleWeights(wavelengths1.z);
    vec3 col = i0.x * w0 + i0.y * w1 + i0.z * w2 + i1.x * w3 + i1.y * w4 + i1.z * w5;

    return interference_gamma(col);
}

vec3 interference(vec3 ray, vec3 normal, float thickness, float roughness) {
    const vec3 wavelengths0 = INTERFERENCE_WAVELENGTHS0;
    const vec3 wavelengths1 = INTERFERENCE_WAVELENGTHS1;
    const vec3 iors0 = (INTERFERENCE_IOR + wavelengths0 * INTERFERENCE_DISPERSION);
    const vec3 iors1 = (INTERFERENCE_IOR + wavelengths1 * INTERFERENCE_DISPERSION);

    vec3 att0 = interference_attenuation(wavelengths0, normal, ray, thickness) * 0.5;
    vec3 att1 = interference_attenuation(wavelengths1, normal, ray, thickness) * 0.5;

    vec3 rray = reflect(ray, normal);
    vec3 cube0 = INTERFERENCE_GAMMA_SCALE * att0 * interference_sampleCubeMap(wavelengths0, rray, roughness, 0.0);
    vec3 cube1 = INTERFERENCE_GAMMA_SCALE * att1 * interference_sampleCubeMap(wavelengths1, rray, roughness, 0.0);

    vec3 f0 = (1.0 - INTERFERENCE_FRESNEL_RATIO) + INTERFERENCE_FRESNEL_RATIO * interference_fresnel(ray, normal, 1.0 / iors0);
    vec3 f1 = (1.0 - INTERFERENCE_FRESNEL_RATIO) + INTERFERENCE_FRESNEL_RATIO * interference_fresnel(ray, normal, 1.0 / iors1);
    vec3 i0 = INTERFERENCE_SCALE * interference_gamma_inverse(cube0 * f0);
    vec3 i1 = INTERFERENCE_SCALE * interference_gamma_inverse(cube1 * f1);

    vec3 rds[6];
    rds[0] = refract(ray, normal, iors0.x);
    rds[1] = refract(ray, normal, iors0.y);
    rds[2] = refract(ray, normal, iors0.z);
    rds[3] = refract(ray, normal, iors1.x);
    rds[4] = refract(ray, normal, iors1.y);
    rds[5] = refract(ray, normal, iors1.z);

    i0 += interference_gamma_inverse( interference_sampleCubeMap(wavelengths0, rds[0], rds[1], rds[2], roughness, 0.0) );
    i1 += interference_gamma_inverse( interference_sampleCubeMap(wavelengths1, rds[3], rds[4], rds[5], roughness, 0.0) );

    vec3 w0 = interference_sampleWeights(wavelengths0.x);
    vec3 w1 = interference_sampleWeights(wavelengths0.y);
    vec3 w2 = interference_sampleWeights(wavelengths0.z);
    vec3 w3 = interference_sampleWeights(wavelengths1.x);
    vec3 w4 = interference_sampleWeights(wavelengths1.y);
    vec3 w5 = interference_sampleWeights(wavelengths1.z);
    vec3 col = i0.x * w0 + i0.y * w1 + i0.z * w2 + i1.x * w3 + i1.y * w4 + i1.z * w5;

    return interference_gamma(col);
}

const int bands = 5;
const float f1 = 0.5; // 1st reflection
const float f2 = 1.0; // 2nd reflection

vec2 interference_light(float w, float s) {
    s *= 2.0*PI/w;
    return vec2(cos(s), sin(s));
}

float interference(float w, float wd, float h) {
    float tot = 0.0;
    for (int i=-bands ; i<=bands ; i++) {
        float id = float(i)/float(bands);
        float cw = w + wd * id;
        
        vec2 l = vec2(0.0); // light/phase
        float f = 1.0; // alpha

        // 1st, distance = 0  , shift = PI
        l += -interference_light(cw, 0.5 * h) * f * f1;
        f *= 1.0-f1;

        // 2nd, distance = 2*h, shift = 0
        l += +interference_light(cw, 2.0 * h) * f * f2;
        f *= 1.0-f2;

        float sensitivity = cos(id * PI)+1.0;
        tot += sensitivity * dot(l, l) / float(bands*2+1);
    }
    return tot;
}

vec3 interference(float h) {
    return vec3(
        interference(650e-9, 60e-9, h),
        interference(532e-9, 40e-9, h),
        interference(441e-9, 30e-9, h)
    );
}

#endif