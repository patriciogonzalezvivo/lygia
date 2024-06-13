/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <float4x4> rotate4dY(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE4DY
#define FNC_ROTATE4DY
float4x4 rotate4dY(in float theta){
    return float4x4(float4(cos(theta),0.,-sin(theta),0),
                    float4(0.,1.,0.,0.),
                    float4(sin(theta),0.,cos(theta),0.),
                    float4(0.,0.,0.,1.));
}
#endif
