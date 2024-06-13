/*
contributors: Patricio Gonzalez Vivo
description: absolute of integer
use: absi(<int> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ABSI
#define FNC_ABSI
#define absi(x)     ( (x < 0)? x * -1 : x )
#endif