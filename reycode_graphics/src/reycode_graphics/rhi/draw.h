#pragma once

namespace reycode {
    struct Shader;

	struct Vertex_Arena;
	struct Vertex_Buffer;

	enum Draw_Mode {
		DRAW_TRIANGLES,
		DRAW_LINE,
		DRAW_POINTS,
		DRAW_MODE_COUNT
	};

    /*struct Command_Buffer {
        void draw(Draw_Mode mode, const Vertex_Buffer&, const Shader&);
    };*/
}