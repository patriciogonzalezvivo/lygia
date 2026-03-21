fn envBRDFApprox(_NoV: f32, _roughness: f32) -> vec2f {
    let c0 = vec4f( -1.0, -0.0275, -0.572, 0.022 );
    let c1 = vec4f( 1.0, 0.0425, 1.04, -0.04 );
    let r = _roughness * c0 + c1;
    let a004 = min( r.x * r.x, exp2( -9.28 * _NoV ) ) * r.x + r.y;
    let AB = vec2f( -1.04, 1.04 ) * a004 + r.zw;
    return vec2f(AB.x, AB.y);
}

//https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
fn envBRDFApprox3(_specularColor: vec3f, _NoV: f32, _roughness: f32) -> vec3f {
    let AB = envBRDFApprox(_NoV, _roughness);
    return _specularColor * AB.x + AB.y;
}

fn envBRDFApproxa(shadingData: ShadingData) -> vec3f {
    return envBRDFApprox(shadingData.specularColor, shadingData.NoV, shadingData.roughness);
}
