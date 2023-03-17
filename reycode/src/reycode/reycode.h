#pragma once

#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <new>
#include <memory>
#include <cmath>
#include <cstdio>

#ifdef _WIN32
#include <intrin.h>
#endif

#ifdef __linux__
#include <xmmintrin.h>
#endif

namespace reycode {
#undef max
#undef min

#ifdef _MSC_VER
#define INLINE inline __forceinline
#else
#define INLINE inline __attribute__((always_inline))
#endif

#ifdef __CUDACC__
#define CGPU __host__ __device__
#else
#define CGPU
#endif

#define INL_CGPU INLINE CGPU
#define INL_CGPU_CONST INLINE CGPU constexpr

    INL_CGPU_CONST size_t kb(size_t n) { return n * 1024; }
    INL_CGPU_CONST size_t mb(size_t n) { return n * 1024 * 1024; }
    INL_CGPU_CONST size_t gb(size_t n) { return n * 1024 * 1024 * 1024; }

    INL_CGPU_CONST uint32_t ceil_div(uint32_t num, uint32_t div) {
        return uint32_t(int32_t(num) + int32_t(div) - int32_t(1)) / div;
    }

    INL_CGPU_CONST uint64_t ceil_div(uint64_t num, uint64_t div) {
        return size_t((int64_t)(num) + (int64_t)(div) - (int64_t)(1)) / div;
    }

    INLINE uint32_t log2i(uint32_t x) {
        unsigned long index = 0;
#ifdef _WIN32
        _BitScanReverse(&index, x);
#endif
#ifdef __linux__
        index = __builtin_ctz(x);
#endif
        return index;
    }

    inline uint32_t pow2i(uint32_t x) {
        return x * x;
    }

    template<class T>
    INL_CGPU_CONST void swap(T& a, T& b) {
        T tmp = a;
        b = a;
        a = tmp;
    }

    template<class T>
    INL_CGPU_CONST T abs(T a) { return a < 0 ? -a : a; }

    template<class T>
    INL_CGPU_CONST T min(T a, T b) { return a < b ? a : b; }

    template<class T>
    INL_CGPU_CONST T max(T a, T b) { return a > b ? a : b; }

    template<class T>
    INL_CGPU_CONST T clamp(T a, T low, T high) { return min(max(a, low), high); }

#ifdef _MSC_VER
#define INLINE inline __forceinline
#else
#define INLINE inline __attribute__((always_inline))
#endif

#define ASSERT_MESG(cond, mesg) if (!(cond)) { fprintf(stderr, "%s: %i", mesg, __LINE__); abort(); }

#define FN1(a, res) [&](auto a) { return res; }
#define FN2(a, b, res) [&](auto a, auto b) { return res; }

    using real = float;

    INL_CGPU constexpr real operator ""_R(long double x) {
        return real(x);
    }

    template<class Func>
    struct Defer {
        Func& func;

        Defer(Func& func) : func(func) {}
        ~Defer() { func(); }
    };

#define SWAP(a, b) { auto tmp = a; a = b; b = tmp; }

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a ## b
#define DEFER(body) auto CONCAT(defer_func,__LINE__) = [&]() { body; }; Defer<decltype(CONCAT(defer_func,__LINE__))> CONCAT(defer,__LINE__) (CONCAT(defer_func,__LINE__))

    struct uvec3 {
        uint32_t x, y, z;

        INL_CGPU_CONST uvec3() : x(0), y(0), z(0) {}
        INL_CGPU_CONST uvec3(uint32_t x, uint32_t y, uint32_t z) : x(x), y(y), z(z) {}
        INL_CGPU_CONST uvec3(const uvec3& vec) : x(vec.x), y(vec.y), z(vec.z) {}
        INL_CGPU_CONST explicit uvec3(uint32_t a) : x(a), y(a), z(a) {}
        INL_CGPU_CONST explicit uvec3(const struct ivec3& a);

#ifdef REY_PLATFORM_CUDA
        INL_CGPU_CONST uvec3(uint3 a) : x(a.x), y(a.y), z(a.z) {}
        INL_CGPU operator dim3() { return { x,y,z }; }
        INL_CGPU operator uint3() { return { x,y,z }; }
#endif

        INL_CGPU uint32_t& operator()(uint32_t i) { return (&x)[i]; }
        INL_CGPU uint32_t& operator[](uint32_t i) { return (&x)[i]; }
    };

    INL_CGPU bool operator<=(uvec3 a, uvec3 b) {
        return a.x <= b.x && a.y <= b.y && a.z <= b.z;
    }

    INL_CGPU bool operator<(uvec3 a, uvec3 b) {
        return a.x < b.x && a.y < b.y && a.z < b.z;
    }

    INL_CGPU bool operator==(uvec3 a, uvec3 b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    INL_CGPU bool operator!=(uvec3 a, uvec3 b) {
        return a.x != b.x || a.y != b.y || a.z != b.z;
    }

    struct ivec3 {
        int32_t x, y, z;

        INL_CGPU_CONST ivec3() : x(0), y(0), z(0) {}
        INL_CGPU_CONST ivec3(int32_t x, int32_t y, int32_t z) : x(x), y(y), z(z) {}
        INL_CGPU_CONST ivec3(uvec3 a) : x(a.x), y(a.y), z(a.z) {}
        INL_CGPU_CONST explicit ivec3(int32_t a) : x(a), y(a), z(a) {}

        INL_CGPU int32_t& operator()(int32_t i) { return (&x)[i]; }
        INL_CGPU int32_t& operator[](int32_t i) { return (&x)[i]; }
    };

    constexpr ivec3 DIR_X{ 1, 0, 0 };
    constexpr ivec3 DIR_Y{ 0, 1, 0 };
    constexpr ivec3 DIR_Z{ 0, 0, 1 };
    constexpr ivec3 DIRS[3] = { DIR_X,DIR_Y,DIR_Z };

    INL_CGPU bool operator<=(ivec3 a, ivec3 b) {
        return a.x <= b.x && a.y <= b.y && a.z <= b.z;
    }

    INL_CGPU bool operator<(ivec3 a, ivec3 b) {
        return a.x < b.x&& a.y < b.y&& a.z < b.z;
    }

    INL_CGPU bool operator==(ivec3 a, ivec3 b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    INL_CGPU bool operator!=(ivec3 a, ivec3 b) {
        return a.x != b.x || a.y != b.y || a.z != b.z;
    }

    INL_CGPU_CONST uvec3::uvec3(const ivec3& a) : x((uint32_t)a.x), y((uint32_t)a.y), z((uint32_t)a.z) {}

    INL_CGPU uvec3 ceil_div(uvec3 a, uvec3 b) {
        return { ceil_div(a.x,b.x), ceil_div(a.y, b.y), ceil_div(a.z, b.z) };
    }

    INL_CGPU uvec3 operator+(uvec3 a, uvec3 b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    INL_CGPU uvec3 operator-(uvec3 a, uvec3 b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    INL_CGPU uvec3 operator*(uint32_t a, uvec3 b) {
        return { a * b.x, a * b.y, a * b.z };
    }

    INL_CGPU uvec3 operator*(uvec3 a, uvec3 b) {
        return { a.x * b.x, a.y * b.y, a.z * b.z };
    }

    INL_CGPU ivec3 operator-(ivec3 a) {
        return { -a.x, -a.y, -a.z };
    }

    INL_CGPU ivec3 operator+(ivec3 a, ivec3 b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    INL_CGPU ivec3 operator-(ivec3 a, ivec3 b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    INL_CGPU ivec3 operator*(int32_t a, ivec3 b) {
        return { a * b.x, a * b.y, a * b.z };
    }

    INL_CGPU ivec3 operator*(ivec3 a, ivec3 b) {
        return { a.x * b.x, a.y * b.y, a.z * b.z };
    }

    using size3 = uvec3;

    struct vec2 {
        real x, y;

        CGPU constexpr vec2() : x(0.0f), y(0.0f) {}
        CGPU constexpr vec2(real x, real y) : x(x), y(y) {}
        CGPU constexpr explicit vec2(real x) : x(x), y(x) {}
        CGPU constexpr explicit vec2(const struct uvec2& x);
        CGPU inline real& operator()(uint32_t i) { return (&this->x)[i]; }
    };

    CGPU inline
    vec2 operator-(vec2 a) {
        return { -a.x, -a.y };
    }

    INL_CGPU_CONST
    vec2 operator+(vec2 a, vec2 b) {
        return { a.x + b.x, a.y + b.y };
    }

    INL_CGPU_CONST
    vec2 operator-(vec2 a, vec2 b) {
        return { a.x - b.x, a.y - b.y };
    }

    INL_CGPU_CONST
    vec2 operator*(real a, vec2 b) {
        return { a * b.x, a * b.y };
    }

    INL_CGPU_CONST
    vec2 operator*(vec2 a, vec2 b) {
        return { a.x * b.x, a.y * b.y };
    }

    INL_CGPU_CONST
    vec2 operator/(vec2 a, vec2 b) {
        return { a.x / b.x, a.y / b.y };
    }

    INL_CGPU_CONST
    void operator+=(vec2& a, vec2 b) {
        a = a + b;
    }

    INL_CGPU_CONST
    void operator-=(vec2& a, vec2 b) {
        a = a - b;
    }

    INLINE CGPU
    void operator*=(vec2& a, vec2 b) {
        a = a * b;
    }

    INL_CGPU_CONST
    void operator/=(vec2& a, vec2 b) {
        a = a / b;
    }

    INL_CGPU_CONST
    real dot(vec2 a, vec2 b) {
        return a.x * b.x + a.y * b.y;
    }

    INL_CGPU_CONST
    real avg(vec2 a) {
        return 0.5_R * (a.x + a.y);
    }

    INL_CGPU real length(vec2 v) {
        return sqrt(v.x * v.x + v.y * v.y);
    }

    INL_CGPU_CONST real sq(vec2 a) {
        return a.x * a.x + a.y * a.y;
    }

    struct vec3 {
        real x, y, z;

        INL_CGPU_CONST vec3() : x(0.0f), y(0.0f), z(0.0f) {}
        INL_CGPU_CONST vec3(real x, real y, real z) : x(x), y(y), z(z) {}
        INL_CGPU_CONST explicit vec3(real x) : x(x), y(x), z(x) {}
        INL_CGPU_CONST vec3(vec2 xy, real z) : x(xy.x), y(xy.y), z(z) {}
        INL_CGPU_CONST explicit vec3(uvec3 a) : x(real(a.x)), y(real(a.y)), z(real(a.z)) {}
        INL_CGPU_CONST explicit vec3(ivec3 a) : x(real(a.x)), y(real(a.y)), z(real(a.z)) {}

        INL_CGPU real& operator()(uint32_t i) { return (&this->x)[i]; }
        INL_CGPU real& operator[](uint32_t i) { return (&this->x)[i]; }

        INL_CGPU_CONST vec2 xy() {
            return { x, y };
        }
    };

    template<uint32_t W>
    struct vec3_wide {
        static constexpr uint32_t WIDTH = W;

        real x[WIDTH];
        real y[WIDTH];
        real z[WIDTH];

        struct accesor {
            vec3_wide& self;
            uint32_t i;

            INL_CGPU_CONST operator vec3() {
                return { self.x[i],self.y[i],self.z[i] };
            }

            INL_CGPU_CONST void operator=(vec3 v) {
                self.x[i] = v.x;
                self.y[i] = v.y;
                self.z[i] = v.z;
            }
        };

        INL_CGPU_CONST vec3 operator[](uint32_t i) const {
            return { x[i], y[i], z[i] };
        }

        INL_CGPU_CONST accesor operator[](uint32_t i) {
            return { *this, i };
        }
    };

    constexpr uint32_t WARP = 32;
    using vec3_warp = vec3_wide<WARP>;

    struct vec4 {
        real x, y, z, w;

        INL_CGPU_CONST vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
        INL_CGPU_CONST vec4(real x, real y, real z, real w) : x(x), y(y), z(z), w(w) {}
        INL_CGPU_CONST vec4(real x) : x(x), y(x), z(x), w(x) {}
        INL_CGPU_CONST vec4(vec3 xyz, real w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}

        INL_CGPU_CONST real& operator()(uint32_t i) { return (&this->x)[i]; }
        INL_CGPU_CONST real& operator[](uint32_t i) { return (&this->x)[i]; }

        INL_CGPU_CONST vec3 xyz() {
            return { x, y, z };
        }
    };

    INL_CGPU_CONST vec4 operator+(vec4 a, vec4 b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
    }

    INL_CGPU_CONST vec4 operator-(vec4 a, vec4 b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    }

    INL_CGPU_CONST vec4 operator*(real a, vec4 b) {
        return { a * b.x, a * b.y, a * b.z, a * b.w };
    }

    struct ivec2 {
        int x, y;

        INL_CGPU_CONST ivec2() : x(0), y(0) {}
        INL_CGPU_CONST ivec2(int x, int y = 1) : x(x), y(y) {}
    };

    struct uvec2 {
        uint32_t x, y;

        INL_CGPU_CONST uvec2() : x(0), y(0) {}
        INL_CGPU_CONST explicit uvec2(uint32_t x) : x(x), y(x) {}
        INL_CGPU_CONST uvec2(uint32_t x, uint32_t y) : x(x), y(y) {}

#ifdef REY_PLATFORM_CUDA
        INL_CGPU_CONST explicit uvec2(uint3 a) : x(a.x), y(a.y) {}
#endif
    };

    CGPU constexpr vec2::vec2(const uvec2& x) : x(real(x.x)), y(real(x.y)) {}

    INL_CGPU_CONST uvec2 operator+(uvec2 a, uvec2 b) {
        return { a.x + b.x, a.y + b.y };
    }

    INL_CGPU_CONST uvec2 operator*(uvec2 a, uvec2 b) {
        return { a.x * b.x, a.y * b.y };
    }

    INL_CGPU_CONST uvec2 operator*(uint32_t a, uvec2 b) {
        return { a * b.x, a * b.y };
    }

    INL_CGPU_CONST ivec2 operator+(ivec2 a, ivec2 b) {
        return { a.x + b.x, a.y + b.y };
    }

    INL_CGPU_CONST uvec2 operator+(uvec2 a, ivec2 b) {
        return { a.x + b.x, a.y + b.y };
    }

    INL_CGPU_CONST vec3 operator+(vec3 a, vec3 b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    INL_CGPU_CONST vec3 operator-(vec3 a, vec3 b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    INL_CGPU_CONST vec3 operator-(vec3 a) {
        return { -a.x, -a.y, -a.z };
    }

    INL_CGPU_CONST vec3 operator*(vec3 a, vec3 b) {
        return { a.x * b.x, a.y * b.y, a.z * b.z };
    }

    INL_CGPU_CONST vec3 operator*(real a, vec3 b) {
        return { a * b.x, a * b.y, a * b.z };
    }

    INL_CGPU_CONST vec3 operator*(vec3 a, real b) {
        return { a.x * b, a.y * b, a.z * b };
    }

    INL_CGPU_CONST vec3 operator/(vec3 a, vec3 b) {
        return { a.x / b.x, a.y / b.y, a.z / b.z };
    }

    INL_CGPU_CONST vec3 operator/(vec3 a, real b) {
        return { a.x / b, a.y / b, a.z / b };
    }

    INL_CGPU_CONST bool operator!=(vec3 a, vec3 b) {
        return a.x != b.x || a.y != b.y || a.z != b.z;
    }

    INL_CGPU_CONST bool operator==(vec3 a, vec3 b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    INL_CGPU_CONST bool operator>(vec3 a, vec3 b) {
        return a.x > b.x && a.y > b.y && a.z > b.z;
    }

    INL_CGPU_CONST bool operator>=(vec3 a, vec3 b) {
        return a.x >= b.x && a.y >= b.y && a.z >= b.z;
    }

    INL_CGPU_CONST bool operator<(vec3 a, vec3 b) {
        return a.x < b.x&& a.y < b.y&& a.z < b.z;
    }

    INL_CGPU_CONST void operator+=(vec3& a, vec3 b) {
        a = a + b;
    }

    INL_CGPU_CONST void operator-=(vec3& a, vec3 b) {
        a = a - b;
    }

    INL_CGPU_CONST void operator*=(vec3& a, vec3 b) {
        a = a * b;
    }

    INL_CGPU_CONST void operator/=(vec3& a, vec3 b) {
        a = a / b;
    }

    INL_CGPU real length(vec3 v) {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    INL_CGPU_CONST real dot(vec3 a, vec3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    INL_CGPU_CONST real sq(vec3 a) {
        return a.x * a.x + a.y * a.y + a.z * a.z;
    }

    INL_CGPU vec3 normalize(vec3 v) {
        return v / length(v);
    }

    INL_CGPU_CONST vec3 cross(vec3 a, vec3 b) {
        return {
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x
        };
    }

    INL_CGPU_CONST vec3 lerp(vec3 a, vec3 b, real f) {
        return a * (1.0_R - f) + b * f;
    }

    INL_CGPU_CONST vec4 lerp(vec4 a, vec4 b, real f) {
        return (1.0_R - f)*a + f*b;
    }

    INL_CGPU_CONST vec3 proj(vec3 vec, vec3 base) {
        return base * dot(vec, base) / sq(base);
    }

    INL_CGPU vec3 abs(vec3 vec) {
        return vec3(::abs(vec.x), ::abs(vec.y), ::abs(vec.z));
    }

    INL_CGPU_CONST vec3 rgb(real r, real g, real b) {
        return (1.0_R / 255.0_R) * vec3(r, g, b);
    }

    INL_CGPU_CONST real max(vec3 vec) {
        if (vec.x > vec.y) return vec.x > vec.z ? vec.x : vec.z;
        else return vec.y > vec.z ? vec.y : vec.z;
    }

    INL_CGPU_CONST uint32_t max(uvec3 vec) {
        if (vec.x > vec.y) return vec.x > vec.z ? vec.x : vec.z;
        else return vec.y > vec.z ? vec.y : vec.z;
    }

    constexpr real PI = (real)3.1415926535897932384626433832795028;

    //todo: move into seperate file
    inline real to_radians(real angle) {
        return angle * real(PI) / 180;
    }

    inline real to_degrees(real angle) {
        return angle * 180 / real(PI);
    }

    //todo move to seperate file
    inline CGPU real vec_angle_cos(vec3 v0, vec3 v1, vec3 v2) {
        vec3 v01 = v0 - v1;
        vec3 v21 = v2 - v1;
        return clamp(dot(v01, v21) / (length(v01) * length(v21)), -1.0_R, 1.0_R);
    }

    inline CGPU real vec_angle(vec3 v0, vec3 v1, vec3 v2) {
        return acos(vec_angle_cos(v0, v1, v2));
    }

    inline CGPU real vec_sign_angle(vec3 v0, vec3 v1, vec3 vn) {
        real angle = acos(clamp(dot(v0, v1) / (length(v0) * length(v1)), -1.0_R, 1.0_R));

        vec3 v2 = cross(-v0, v1);
        if (dot(vn, v2) < -FLT_EPSILON) { // Or > 0
            angle = -angle;
        }
        return angle;
    }

    inline real vec_dir_angle(vec3 v0, vec3 v1, vec3 vn) {
        real angle = vec_sign_angle(v0, v1, vn);
        if (angle < 0) {
            angle = 2 * real(PI) + angle;
        }
        return angle;
    }

    inline real vec_angle(vec3 v0, vec3 v1) {
        return acos(clamp(dot(v0, v1) / (length(v0) * length(v1)), -1.0_R, 1.0_R));
    }

    struct mat4x4 {
        real data[4][4] = {};

        INL_CGPU mat4x4() {}
        mat4x4(std::initializer_list<real> list) {
            assert(list.size() == 16);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) data[i][j] = list.begin()[4 * i + j];
            }
        }

        INL_CGPU mat4x4(real diagonal) {
            for (int i = 0; i < 4; i++) data[i][i] = diagonal;
        }

        INL_CGPU real& operator()(uint32_t i, uint32_t j) {
            return data[i][j];
        }

        INL_CGPU real operator()(uint32_t i, uint32_t j) const {
            return data[i][j];
        }
    };

    INLINE vec4 operator*(mat4x4 a, vec4 b) {
        vec4 result;
        for (int i = 0; i < 4; i++) {
            real sum = 0.0_R;
            for (int j = 0; j < 4; j++) sum += a(i, j) * b(j);
            result(i) = sum;
        }
        return result;
    }

    INLINE mat4x4 operator*(const mat4x4& a, const mat4x4& b) {
        mat4x4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                real sum = 0.0_R;
                for (int p = 0; p < 4; p++) {
                    sum += a(i, p) * b(p, j);
                }
                result(i,j) = sum;
            }
        }
        return result;
    }

    INLINE mat4x4 translate_mat(vec3 x) {
        mat4x4 mat(1.0);
        mat(0, 3) += x.x;
        mat(1, 3) += x.y;
        mat(2, 3) += x.z;
        return mat;
    }

    INLINE mat4x4 rotate_x(real r) {
        return {
                1,0,0,0,
                0,cos(r),sin(r),0,
                0,-sin(r),cos(r),0,
                0,0,0,1
        };
    }

    INLINE mat4x4 rotate_y(real r) {
        return {
                cos(r),0,-sin(r),0,
                0,1,0,0,
                sin(r),0,cos(r),0,
                0,0,0,1
        };
    }

    INLINE mat4x4 rotate_z(real r) {
        return {
                cos(r),-sin(r),0,0,
                sin(r),cos(r),0,0,
                0,0,1,0,
                0,0,0,1
        };
    }

    INLINE mat4x4 rotate_mat(vec3 r) {
        return rotate_x(r.x) * rotate_y(r.y) * rotate_z(r.z);
    }

    INLINE mat4x4 projection_matrix(real l, real r, real b, real t, real n, real f) {
        return {
                2*n/(r-l), 0, (r+l)/(r-l), 0,
                0, 2*n/(t-b), (t+b)/(t-b), 0,
                0, 0, -(f+n)/(f-n), -2*f*n/(f-n),
                0, 0, -1, 0
        };
    }

    INLINE mat4x4 projection_matrix(real aspect, real fov, real n, real f) {
        real r = n * tan(fov/2) * aspect;
        real t = n * tan(fov/2);
        return projection_matrix(-r, r, -t, t, n, f);
    }

    struct mat2x2 {
        real data[2][2] = {};

        INL_CGPU mat2x2(float diagonal) {
            for (int i = 0; i < 2; i++) data[i][i] = diagonal;
        }

        INL_CGPU real& operator()(uint32_t i, uint32_t j) {
            return data[i][j];
        }
    };

    template<class T, uint32_t N>
    struct array {
        T data[N];
        uint32_t length;

        INL_CGPU array() {}
        INL_CGPU array(uint32_t length) : length(length) {}
        INL_CGPU array(std::initializer_list<T> list) : data() {
            assert(list.size() <= N);
            length = 0;
            for (T value : list) data[length++] = value;
        }

        INL_CGPU T& operator[](uint32_t i) {
#ifdef BOUNDS_CHECKING
            assert(i < length);
#endif
            return data[i];
        }

        INL_CGPU const T& operator[](uint32_t i) const {
#ifdef BOUNDS_CHECKING
            assert(i < length);
#endif
            return data[i];
        }

        INL_CGPU const T* begin() const { return data; }
        INL_CGPU const T* end() const { return data + length; }
        INL_CGPU T* begin() { return data; }
        INL_CGPU T* end() { return data + length; }
        uint32_t size() const { return length; }
    };

    template<class T>
    struct slice {
        T* data;
        uint32_t length;

        INL_CGPU slice() : data(nullptr), length(0) {}
        INL_CGPU slice(T& value) : data(&value), length(1) {}
        //slice(std::initializer_list<T> list) : data((T*)list.begin()), length(list.size()) {} //todo: introduce const slice
        INL_CGPU slice(T* data, uint32_t length) : data(data), length(length) {}

        INL_CGPU T& operator[](uint32_t i) {
#ifdef BOUNDS_CHECKING
            assert(i < length);
#endif
            return data[i];
        }

        INL_CGPU const T& operator[](uint32_t i) const {
#ifdef BOUNDS_CHECKING
            assert(i < length);
#endif
            return data[i];
        }

        INL_CGPU T* begin() { return data; }
        INL_CGPU T* end() { return data + length; }
        INL_CGPU const T* begin() const { return data; }
        INL_CGPU const T* end() const { return data + length; }
    };

    template<class T>
    struct slice3 {
        union {
            struct {
                slice<T> x;
                slice<T> y;
                slice<T> z;
            };
            slice<T> data[3];
        };
    };

    template<class T, uint32_t X, uint32_t Y>
    struct array2d {
        T data[Y][X];


        INL_CGPU T& operator()(uvec2 idx) {
            assert(idx.x < X);
            assert(idx.y < Y);
            return data[idx.y][idx.x];
        }

        INL_CGPU T& operator()(int x, int y) {
            assert(0 <= x && x < X);
            assert(0 <= y && y < Y);
            return data[y][x];
        }
    };

    template<>
    struct slice<vec3_warp> {
        vec3_warp* data;
        uint32_t length;

        INL_CGPU slice() : data(nullptr), length(0) {}
        INL_CGPU slice(vec3_warp& value) : data(&value), length(1) {}
        //slice(std::initializer_list<T> list) : data((T*)list.begin()), length(list.size()) {} //todo: introduce const slice
        INL_CGPU slice(vec3_warp* data, uint32_t length) : data(data), length(length) {}

        INL_CGPU vec3_warp::accesor operator[](uint32_t i) {
#ifdef BOUNDS_CHECKING
            assert(i < WARP*length);
#endif
            return data[i/WARP][i%WARP];
        }

        INL_CGPU const vec3_warp::accesor operator[](uint32_t i) const {
#ifdef BOUNDS_CHECKING
            assert(i < WARP*length);
#endif
            return data[i/WARP][i%WARP];
        }

        INL_CGPU vec3_warp* begin() { return data; }
        INL_CGPU vec3_warp* end() { return data + length; }
        INL_CGPU const vec3_warp* begin() const { return data; }
        INL_CGPU const vec3_warp* end() const { return data + length; }
    };

    struct range {
        uint32_t base;
        uint32_t length;
        uint32_t capacity;
    };

    template<class T>
    slice<T> INL_CGPU subslice(slice<T> s, uint32_t base, uint32_t length) {
#ifdef BOUNDS_CHECKING
        assert(base <= s.length);
		assert(base + length <= s.length);
#endif
        return { s.data + base, length };
    }

    template<class T>
    slice<T> INL_CGPU subslice(slice<T> s, range r) {
        return subslice(s, r.base, r.length);
    }

    struct Arena {
        uint8_t* data;
        size_t capacity;
        size_t used;
    };

    Arena make_host_arena(uint64_t size);
    void destroy_host_arena(Arena& arena);

    INLINE void arena_reset(Arena& arena, size_t size) {
        arena.used = size;
    }

    INLINE void* arena_push(Arena& arena, size_t size, size_t align = 1) {
        size_t offset = align * ceil_div((uint64_t)arena.used, align);

        void* ptr = arena.data + offset;
        arena.used = offset + size;
        assert(arena.used < arena.capacity);
        return ptr;
    }

    template<class T>
    INLINE T* arena_push(Arena& arena, size_t align = 0) {
        return (T*)reycode::arena_push(arena, sizeof(T), align == 0 ? alignof(T) : align);
    }

    template<class T>
    INLINE T* arena_push(Arena& arena, T&& init, size_t align = 0) {
        T* ptr = (T*)reycode::arena_push(arena, sizeof(T), align == 0 ? alignof(T) : align);
        new (ptr) T(std::move(init));
        return ptr;
    }

    template<class T>
    INLINE slice<T> arena_push_array(Arena& arena, size_t length, size_t align = 0) {
        slice<T> result = {};
        result.length = length;
        result.data = (T*)arena_push(arena, length * sizeof(T), align == 0 ? alignof(T) : align);
        return result;
    }

    struct LinearRegion {
        Arena& arena;
        uint64_t watermark;

        LinearRegion(Arena& arena) : arena(arena) {
            watermark = arena.used;
        }

        ~LinearRegion() {
            arena_reset(arena, watermark);
        }
    };

#ifdef PLATFORM_CUDA
    template<class T>
    __declspec(noinline) void debug_view_device_values(slice<T> device, uint32_t count) {
        assert(count < 1024);
        static T host[1024];
        cudaMemcpy(host, device.data, sizeof(T) * count, cudaMemcpyDeviceToHost);
        return;
    }
#endif

    struct Memory_Ctx {
        Arena host_perm_arena;
        Arena host_temp_arena;
        Arena device_perm_arena;
        Arena device_temp_arena;
    };

#ifdef REY_PLATFORM_CUDA
    Memory_Ctx make_memory_ctx(uint64_t host_perm, uint64_t host_arena, uint64_t device_perm, uint64_t device_arena, Cuda_Error& err);
    void destroy_memory_ctx(Memory_Ctx&);
#endif
}