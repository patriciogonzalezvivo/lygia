#ifndef FNC_SPECULAR_GAUSSIAN
#define FNC_SPECULAR_GAUSSIAN

// https://github.com/glslify/glsl-specular-gaussian
float specularGaussian(ShadingData shadingData) {
    float theta = acos(shadingData.NoH);
    float w = theta / shadingData.linearRoughness;
    return exp(-w*w);
}

#endif