#ifndef FNC_ENVBRDFAPPROX
#define FNC_ENVBRDFAPPROX

float3 envBRDFApprox(float3 _specularColor, float _NoV, float _roughness) {
    float4 c0 = float4( -1, -0.0275, -0.572, 0.022 );
    float4 c1 = float4( 1, 0.0425, 1.04, -0.04 );
    float4 r = _roughness * c0 + c1;
    float a004 = min( r.x * r.x, exp2( -9.28 * _NoV ) ) * r.x + r.y;
    float2 AB = float2( -1.04, 1.04 ) * a004 + r.zw;
    return _specularColor * AB.x + AB.y;
}

#endif