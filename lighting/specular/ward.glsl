#ifndef FNC_SPECULAR_WARD
#define FNC_SPECULAR_WARD

// https://github.com/glslify/glsl-specular-ward
float specularWard(const in vec3 L, const in vec3 N, const in vec3 V, const in vec3 fiber, const in float shinyParallel, const in float shinyPerpendicular) {
    float NdotL = dot(N, L);
    float NdotR = dot(N, V);
    vec3 fiberParallel = normalize(fiber);
    vec3 fiberPerpendicular = normalize(cross(N, fiber));

    if (NdotL < 0.0 || NdotR < 0.0)
        return 0.0;

    vec3 H = normalize(L + V);

    float NdotH = dot(N, H);
    float XdotH = dot(fiberParallel, H);
    float YdotH = dot(fiberPerpendicular, H);

    float coeff = sqrt(NdotL/NdotR) / (12.5663706144 * shinyParallel * shinyPerpendicular); 
    float theta = (pow(XdotH/shinyParallel, 2.0) + pow(YdotH/shinyPerpendicular, 2.0)) / (1.0 + NdotH);

    return coeff * exp(-2.0 * theta);
}

#endif