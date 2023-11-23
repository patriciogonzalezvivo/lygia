#include "fresnel.wgsl"
#include "envMap.wgsl"

fn fresnelReflection(R: vec3f, f0: vec3f, NoV: f32) -> {
    let frsnl = fresnel(f0, NoV);

    let reflectColor = vec3f(0.0);
    reflectColor = envMap(R, 1.0, 0.001);

    return reflectColor * frsnl;
}