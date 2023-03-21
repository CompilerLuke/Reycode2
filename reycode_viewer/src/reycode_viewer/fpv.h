#pragma once

#include "reycode/reycode.h"

namespace reycode {
    struct Input_State;

    struct FPV {
        real fov = 0.2_R*PI;
        real near = 0.01_R;
        real far = 200;
        vec3 view_pos;

        vec3 forward_dir;
        vec3 right_dir;
        vec3 up_dir;
        real yaw;
        real pitch;
        vec2 prev_cursor_pos;
        bool capture_cursor;

        real mouse_sensitivity = 1.0_R;
    };

    void fpv_update(FPV& camera, const Input_State& input, real dt);
    mat4x4 fpv_view_mat(const FPV& fpv);
    mat4x4 fpv_proj_mat(const FPV& fpv, uvec2 extent);
}
