/*
contributors: Patricio Gonzalez Vivo
description: absolute of integer
use: absi(<int> value)
*/

#ifndef FNC_ABSI
#define FNC_ABSI
#define absi(x)     ( (x < 0)? x * -1 : x )
#endif