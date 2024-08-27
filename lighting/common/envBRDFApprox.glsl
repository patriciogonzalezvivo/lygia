#ifndef FNC_ENVBRDFAPPROX
#define FNC_ENVBRDFAPPROX

vec2 envBRDFApprox(const in float _NoV, in float _roughness ) {
    const vec4 c0 = vec4( -1.0, -0.0275, -0.572, 0.022 );
    const vec4 c1 = vec4( 1.0, 0.0425, 1.04, -0.04 );
    vec4 r = _roughness * c0 + c1;
    float a004 = min( r.x * r.x, exp2( -9.28 * _NoV ) ) * r.x + r.y;
    vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;
    return vec2(AB.x, AB.y);
}

//https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
vec3 envBRDFApprox(const in vec3 _specularColor, const in float _NoV, const in float _roughness) {
    vec2 AB = envBRDFApprox(_NoV, _roughness);
    return _specularColor * AB.x + AB.y;
}

vec3 envBRDFApprox(ShadingData shadingData) {
    return envBRDFApprox(shadingData.specularColor, shadingData.NoV, shadingData.roughness);
}

#endif