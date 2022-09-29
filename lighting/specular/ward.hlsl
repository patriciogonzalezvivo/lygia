#ifndef FNC_SPECULAR_WARD
#define FNC_SPECULAR_WARD

// https://github.com/glslify/glsl-specular-ward
float specularWard(float3 L, float3 N, float3 V, float3 fiber, float shinyParallel, float shinyPerpendicular) {
    float NdotL = dot(N, L);
    float NdotR = dot(N, V);
    float3 fiberParallel = normalize(fiber);
    float3 fiberPerpendicular = normalize(cross(N, fiber));

    if (NdotL < 0.0 || NdotR < 0.0)
        return 0.0;

    float3 H = normalize(L + V);

    float NdotH = dot(N, H);
    float XdotH = dot(fiberParallel, H);
    float YdotH = dot(fiberPerpendicular, H);

    float coeff = sqrt(NdotL/NdotR) / (12.5663706144 * shinyParallel * shinyPerpendicular); 
    float theta = (pow(XdotH/shinyParallel, 2.0) + pow(YdotH/shinyPerpendicular, 2.0)) / (1.0 + NdotH);

    return coeff * exp(-2.0 * theta);
}

#endif