#ifndef FNC_SPECULAR_GAUSSIAN
#define FNC_SPECULAR_GAUSSIAN

// https://github.com/glslify/glsl-specular-gaussian
float specularGaussian(const in vec3 NoH, const in float roughness) {
    float theta = acos(NoH);
    float w = theta / roughness;
    return exp(-w*w);
}

float specularGaussian(ShadingData shadingData) {
    return specularGaussian(shadingData.NoH, shadingData.roughness);
}

#endif