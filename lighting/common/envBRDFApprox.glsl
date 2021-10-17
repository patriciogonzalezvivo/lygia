#ifndef FNC_ENVBRDFAPPROX
#define FNC_ENVBRDFAPPROX

vec3 envBRDFApprox(vec3 _specularColor, float _NoV, float _roughness) {
    vec4 c0 = vec4( -1, -0.0275, -0.572, 0.022 );
    vec4 c1 = vec4( 1, 0.0425, 1.04, -0.04 );
    vec4 r = _roughness * c0 + c1;
    float a004 = min( r.x * r.x, exp2( -9.28 * _NoV ) ) * r.x + r.y;
    vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;
    return _specularColor * AB.x + AB.y;
}

#endif