import lygia::math::saturate::saturate;

/*
contributors: Inigo Quiles
description: Union operation of two SDFs

*/

fn opUnion( d1: f32, d2: f32 ) -> f32 { return min(d1, d2); }

// Soft union
fn opUnionSmooth( d1: f32, d2: f32, k: f32 ) -> f32 {
    let h = saturate( 0.5 + 0.5*(d2-d1)/k );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

fn opUnionSmooth4( d1: vec4f, d2: vec4f, k: f32 ) -> vec4f {
    let h = saturate( 0.5 + 0.5*(d2.a - d1.a)/k );
    var result = mix(d2, d1, h);
    result.a -= k * h * (1.0 - h);
    return result;
}
