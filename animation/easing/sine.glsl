#include "../../math/const.glsl"

/*
author: Hugh Kennedy (https://github.com/hughsk)
description: sine easing. From https://github.com/stackgl/glsl-easings
use: sine<In|Out|InOut>(<float> x)
license: |
  This software is released under the MIT license:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_SINEIN
#define FNC_SINEIN
float sineIn(in float t) {
    return sin((t - 1.0) * HALF_PI) + 1.0;
}
#endif

#ifndef FNC_SINEOUT
#define FNC_SINEOUT
float sineOut(in float t) {
    return sin(t * HALF_PI);
}
#endif

#ifndef FNC_SINEINOUT
#define FNC_SINEINOUT
float sineInOut(in float t) {
    return -0.5 * (cos(PI * t) - 1.0);
}
#endif
