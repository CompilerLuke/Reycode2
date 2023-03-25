#pragma once

#include "reycode/reycode.h"

namespace reycode {
    struct Colormap {
        static constexpr uint32_t color_count = 20;
        vec3 colors[color_count];

        static Colormap viridis();
    };

    INL_CGPU vec3 color_map(Colormap cm, real value, real min_value = 0.0, real max_value = 1.0) {
        real delta = max_value - min_value;
        real range = delta > 0 ? max(delta, 1e-5_R) : min(delta, -1e-5_R);

        int n = cm.color_count;

        value = (value - min_value) / delta * n;
        value = roundf(value);

        int a = clamp(int(value), 0, n - 1);
            //max(0, min(value, n - 1));
        int b = min(a + 1, n - 1);
        return lerp(cm.colors[a], cm.colors[b], value - int(value));
    }
}