const SRGB2XYZ = mat3x3<f32>(   vec3f( 0.4124564, 0.2126729, 0.0193339),
                                vec3f( 0.3575761, 0.7151522, 0.1191920),
                                vec3f( 0.1804375, 0.0721750, 0.9503041) );
fn srgb2xyz(srgb: vec3f) -> vec3f { return SRGB2XYZ * srgb;}