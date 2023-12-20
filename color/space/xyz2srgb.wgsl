const XYZ2SRGB = mat3x3<f32>(   vec3f( 3.2404542, -0.9692660,  0.0556434),
                                    vec3f(-1.5371385,  1.8760108, -0.2040259),
                                    vec3f(-0.4985314,  0.0415560,  1.0572252) );
fn xyz2srgb(xyz: vec3f) -> vec3f { return XYZ2SRGB * srgb;}