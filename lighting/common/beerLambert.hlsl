/*
contributors: Shadi El Hajj
description: The Beer-Lambert law 
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_BEER_LAMBERT
#define FNC_BEER_LAMBERT

float beerLambert(float absorption, float dist) {
    return exp(-absorption * dist);
}

#endif
