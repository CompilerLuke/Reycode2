#include <glad/glad.h>
#define GLFW_NO_INCLUDE
#include <GLFW/glfw3.h>
#include "reycode/reycode.h"
#include "reycode_graphics/rhi/window.h"
#include <string>

namespace reycode {
    void window_error_callback(int error, const char* description) {
        fprintf(stderr, "Window error");
        //fprintf(stderr, "Error (%i) : %s\n", error, description);
    }
    
    void gl_message_callback(
        GLenum source,
        GLenum type,
        GLuint id,
        GLenum severity,
        GLsizei length,
        const GLchar* message,
        const void* userParam) {
        fprintf(stderr, "Warning\n");
        //fprintf(stderr, "GL CALLBACK: type = 0x%x, severity = 0x%x, message = %s\n",
        //    type, severity, message);
    }

    static void key_callback(GLFWwindow* window, int key, int, int action, int) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    Window::Window(const Window_Desc& desc) {
        glfwSetErrorCallback(window_error_callback);

        if (!glfwInit()) throw std::string("Could not initialize glfw");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);

        native = glfwCreateWindow(desc.width, desc.height, desc.title, NULL, NULL);
        if (!native) {
            glfwTerminate();
            throw std::string("Could not create window");
        }

        glfwSetWindowUserPointer(native, nullptr);
        glfwSetKeyCallback(native, key_callback);

        glfwMakeContextCurrent(native);
        gladLoadGL();

        if (desc.vsync) glfwSwapInterval(1);
        else glfwSwapInterval(0);

        if (desc.validation) {
            glEnable(GL_DEBUG_OUTPUT);
            glDebugMessageCallback(gl_message_callback, 0);
        }
    }

    bool Window::is_open() const {
        return !glfwWindowShouldClose(native);
    }

    void Window::draw() {
        glfwSwapBuffers(native);
    }

    Window::~Window() {
        glfwDestroyWindow(native);
        glfwTerminate();
    }

    int KEY_TO_GLFW[KEY_COUNT] = {
        GLFW_KEY_LEFT_SHIFT,
        GLFW_KEY_RIGHT_SHIFT,

        GLFW_KEY_A,
        GLFW_KEY_B,
        GLFW_KEY_C,
        GLFW_KEY_D,
        GLFW_KEY_E,
        GLFW_KEY_F,
        GLFW_KEY_G,
        GLFW_KEY_H,
        GLFW_KEY_I,
        GLFW_KEY_J,
        GLFW_KEY_K,
        GLFW_KEY_L,
        GLFW_KEY_M,
        GLFW_KEY_N,
        GLFW_KEY_O,
        GLFW_KEY_P,
        GLFW_KEY_Q,
        GLFW_KEY_R,
        GLFW_KEY_S,
        GLFW_KEY_T,
        GLFW_KEY_U,
        GLFW_KEY_V,
        GLFW_KEY_W,
        GLFW_KEY_X,
        GLFW_KEY_Y,
        GLFW_KEY_Z,

        GLFW_KEY_RIGHT,
        GLFW_KEY_LEFT,
        GLFW_KEY_DOWN,
        GLFW_KEY_UP,

        GLFW_KEY_SPACE
    };

    int MOUSE_BUTTON_TO_GLFW[MOUSE_BUTTON_COUNT] = {
        GLFW_MOUSE_BUTTON_LEFT,
        GLFW_MOUSE_BUTTON_RIGHT,
        GLFW_MOUSE_BUTTON_MIDDLE
    };


    bool Input_State::mouse_button_down(Mouse_Button button) const {
        return mouse_button_state[button] >= KEY_PRESS;
    }

    bool Input_State::key_down(Key key) const {
        return key_state[key] >= KEY_PRESS;
    }

    bool Input_State::key_pressed(Key key) const {
        return key_state[key] == KEY_PRESS;
    }

    bool Input_State::key_down(Key_Binding key) const {
        return key_binding_state[key] >= KEY_PRESS;
    }

    Input_State Window::poll() {
        glfwPollEvents();

        double xpos, ypos;
        glfwGetCursorPos(native, &xpos, &ypos);

        auto update_state = [](Key_State& state, bool pressed) {
            if (pressed && state < KEY_PRESS)
                state = KEY_PRESS;
            else if (pressed && state == KEY_PRESS)
                state = KEY_REPEAT;
            else if (!pressed && state >= KEY_PRESS)
                state = KEY_RELEASE;
            else if (!pressed && state == KEY_RELEASE)
                state = KEY_UP;
        };

        vec2 cursor_pos = { real(xpos), real(ypos) };
        state.cursor_delta = init ? cursor_pos - state.cursor_pos : vec2(0);
        state.cursor_pos = cursor_pos;

        for (int i = 0; i < MOUSE_BUTTON_COUNT; i++) {
            update_state(state.mouse_button_state[i], glfwGetMouseButton(native, MOUSE_BUTTON_TO_GLFW[i]));
        }

        for (int i = 0; i < KEY_COUNT; i++) {
            update_state(state.key_state[i], glfwGetKey(native, KEY_TO_GLFW[i]));
        }

        state.key_binding_state[KEY_BIND_UP] = state.key_state[KEY_W];
        state.key_binding_state[KEY_BIND_DOWN] = state.key_state[KEY_S];
        state.key_binding_state[KEY_BIND_RIGHT] = state.key_state[KEY_D];
        state.key_binding_state[KEY_BIND_LEFT] = state.key_state[KEY_A];

        init = true;
        return state;
    }

    void Window::capture_cursor(bool capture) {
        glfwSetInputMode(native, GLFW_CURSOR, capture ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }

    double Window::get_time() {
        return glfwGetTime();
    }
}