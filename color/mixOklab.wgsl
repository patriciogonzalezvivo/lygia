fn mixOklab( colA: vec3<f32>, colB: vec3<f32>, h: f32 ) -> vec3<f32> {
    let kCONEtoLMS = mat3x3<f32>(                
        vec3<f32>(0.4121656120,  0.2118591070,  0.0883097947),
        vec3<f32>(0.5362752080,  0.6807189584,  0.2818474174),
        vec3<f32>(0.0514575653,  0.1074065790,  0.6302613616) );

    let kLMStoCONE = mat3x3<f32>(
        vec3<f32>(4.0767245293, -1.2681437731, -0.0041119885),
        vec3<f32>(-3.3072168827,  2.6093323231, -0.7034763098),
        vec3<f32>(0.2307590544, -0.3411344290,  1.7068625689));
                    
    // rgb to cone (arg of pow can't be negative)
    let lmsA = pow( kCONEtoLMS*colA, vec3<f32>(1.0/3.0) );
    let lmsB = pow( kCONEtoLMS*colB, vec3<f32>(1.0/3.0) );

    let lms = mix( lmsA, lmsB, h );
    
    // gain in the middle (no oaklab anymore, but looks better?)
    //lms *= 1.0+0.2*h*(1.0-h);

    // cone to rgb
    return kLMStoCONE*(lms*lms*lms);
}
