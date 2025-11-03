/*
contributors: Patricio Gonzalez Vivo
description: absolute value of integer
use: absi(value: i32) -> i32
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn absi(x: i32) -> i32 {
    return select(x, x * -1, x < 0);
}
