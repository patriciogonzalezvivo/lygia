/*
contributors: Patricio Gonzalez Vivo
description: Blackbody in kelvin to RGB. Range between 0.0 and 40000.0 Kelvin
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn k2rgb(t : f32) -> vec3f {
    let p = pow(t, -1.5);
    let l = log(t);
    var color = vec3f(
        220000.0 * p + 0.5804,
        0.3923 * l - 2.4431,
        0.7615 * l - 5.681
    );

    if (t > 6500.0) 
        color.g = 138039.0 * p + 0.738;

    color = saturate(color);
    if (t < 1000.0) 
        color *= t/1000.0;

    return color;
}
