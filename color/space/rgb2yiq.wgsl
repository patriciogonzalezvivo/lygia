const RGB2YIQ : mat3x3<f32>  = mat3x3<f32>( 
    vec3f(0.300,  0.5900,  0.1100), 
    vec3f(0.599, -0.2773, -0.3217), 
    vec3f(0.213, -0.5251,  0.3121) );

fn rgb2yiq(rgb : vec3f) -> vec3f { return RGB2YIQ * rgb; }
