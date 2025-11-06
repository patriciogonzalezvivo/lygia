import lygia::math::saturate::saturate;

/*
contributors: Inigo Quiles
description: Subtraction operation of two SDFs

*/

fn opSubtraction( d1: f32, d2: f32 ) -> f32 { return max(-d1, d2); }

fn opSubtraction4( d1: vec4f, d2: vec4f ) -> vec4f {
    return select(d2, -d1, -d1.a > d2.a);
}

fn opSubtractionSmooth( d1: f32, d2: f32, k: f32 ) -> f32 {
    let h = saturate( 0.5 - 0.5*(d2+d1)/k );
    return mix( d2, -d1, h ) + k*h*(1.0-h);
}

fn opSubtractionSmooth4( d1: vec4f, d2: vec4f, k: f32 ) -> vec4f {
    let h = saturate( 0.5 - 0.5*(d2.a+d1.a)/k );
    var result = mix( d2, -d1, h );
    result.a += k*h*(1.0-h);
    return result;
}
