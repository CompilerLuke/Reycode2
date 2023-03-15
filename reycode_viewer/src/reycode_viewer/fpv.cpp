#include "reycode/reycode.h"
#include "reycode_viewer/fpv.h"
#include "reycode_graphics/rhi/window.h"

namespace reycode {
    void fpv_update(FPV& camera, const Input_State& input, real dt) {
        vec2 cursor_delta = input.cursor_delta;
        const real sensitivity = 5e-2_R;

        camera.capture_cursor = input.mouse_button_down(MOUSE_BUTTON_RIGHT);

        if (camera.capture_cursor) {
            camera.yaw += dt * sensitivity * cursor_delta.x;
            camera.pitch += dt * sensitivity * cursor_delta.y;
            camera.pitch = clamp(camera.pitch, -PI / 2.0_R, PI / 2.0_R);
        }

        mat4x4 view_to_world = rotate_y(camera.yaw) * rotate_x(camera.pitch);
        vec3 forward = (view_to_world * vec4(0, 0, -1, 0)).xyz();
        vec3 right = (view_to_world * vec4(1, 0, 0, 0)).xyz();
        vec3 up = (view_to_world * vec4(0, 0, 1, 0)).xyz();

        camera.forward_dir = forward;
        camera.right_dir = right;
        camera.up_dir = up;

        real move_speed = 2 * max(1.0_R, camera.view_pos.y);
        if (input.key_down(KEY_LEFT_SHIFT)) move_speed *= 2;
        if (input.key_down(KEY_BIND_UP)) camera.view_pos += move_speed * dt * forward;
        if (input.key_down(KEY_BIND_DOWN)) camera.view_pos += move_speed * dt * -forward;
        if (input.key_down(KEY_BIND_RIGHT)) camera.view_pos += move_speed * dt * right;
        if (input.key_down(KEY_BIND_LEFT)) camera.view_pos += move_speed * dt * -right;
    }

    mat4x4 fpv_view_mat(const FPV& fpv) {
        return rotate_x(-fpv.pitch) * rotate_y(-fpv.yaw) * translate_mat(-fpv.view_pos);
    }

    mat4x4 fpv_proj_mat(const FPV& fpv, uvec2 extent) {
        real near = fpv.near;
        real far = fpv.far;
        real fov = fpv.fov;
        real aspect = real(extent.x) / extent.y;
        return projection_matrix(aspect, fov, near, far);
    }
}