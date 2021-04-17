#include "../../math/const.glsl"

/*
author: Hugh Kennedy (https://github.com/hughsk)
description: Back easing. From https://github.com/stackgl/glsl-easings
use: back<In|Out|InOut>(<float> x)
license: |
  This software is released under the MIT license:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_BACKIN
#define FNC_BACKIN
float backIn(in float t) {
    return pow(t, 3.) - t * sin(t * PI);
}
#endif

#ifndef FNC_BACKOUT
#define FNC_BACKOUT
float backOut(in float t) {
    return 1. - backIn(1. - t);
}
#endif

#ifndef FNC_BACKINOUT
#define FNC_BACKINOUT
float backInOut(in float t) {
    float f = t < .5
        ? 2.0 * t
        : 1.0 - (2.0 * t - 1.0);

    float g = backIn(f);

    return t < 0.5
        ? 0.5 * g
        : 0.5 * (1.0 - g) + 0.5;
}
#endif
