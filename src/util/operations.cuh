#pragma once

#include "consts.h"
#include <math.h>

__device__ __host__
float to_radians(float angle) {
    return angle / 180.0 * consts::PI;
}

__device__ __host__
float to_degrees(float angle) {
    return angle / consts::PI * 180.0;
}

// Float4 Operations
__device__ __host__
float4 operator+(const float4& lhs, const float4& rhs) {
    return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

// Float4 Operations
__device__ __host__
float4& operator+=(float4& lhs, const float4& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

__device__ __host__
float4 operator-(const float4& lhs, const float4& rhs) {
    return make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

__device__ __host__
float4 operator*(const float4& lhs, const float& rhs) {
    return make_float4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}

__device__ __host__
float4 operator*(const float& lhs, const float4& rhs) {
    return make_float4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}

__device__ __host__
float4 operator/(const float4& lhs, const float& rhs) {
    return make_float4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}



// Float3 Operations
__device__ __host__
float3 operator+(const float3& lhs, const float3& rhs) {
    return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__device__ __host__
float3 operator-(const float3& lhs, const float3& rhs) {
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z + rhs.z);
}

__device__ __host__
float3 operator*(const float3& lhs, const float& rhs) {
    return make_float3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

__device__ __host__
float3 operator*(const float& lhs, const float3& rhs) {
    return make_float3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}

__device__ __host__
float3 operator/(const float3& lhs, const float& rhs) {
    return make_float3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}


// Float2 Operations
__device__ __host__
float2 operator+(const float2& lhs, const float2& rhs) {
    return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

__device__ __host__
float2 operator-(const float2& lhs, const float2& rhs) {
    return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

__device__ __host__
float2 operator*(const float2& lhs, const float& rhs) {
    return make_float2(lhs.x * rhs, lhs.y * rhs);
}

__device__ __host__
float2 operator*(const float& lhs, const float2& rhs) {
    return make_float2(lhs * rhs.x, lhs * rhs.y);
}

__device__ __host__
float2 operator/(const float2& lhs, const float& rhs) {
    return make_float2(lhs.x / rhs, lhs.y / rhs);
}

__device__ __host__ 
float perp_dot(const float2& lhs, const float2& rhs) {
    return (lhs.x * rhs.y) - (lhs.y * rhs.x);
}

__device__ __host__ 
float dot(const float2& lhs, const float2& rhs) {
    return (lhs.x * rhs.x) + (lhs.y * rhs.y);
}

__device__ __host__ 
float angle(const float2& lhs, const float2& rhs) {
    return atan2(perp_dot(lhs, rhs), dot(lhs, rhs));
}

__device__ __host__ 
float2 perp(const float2& lhs) {
    return make_float2(lhs.y, -lhs.x);
}


// Conversions
__device__ __host__
float3 vec4_to_vec3(const float4& vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

__device__ __host__
float2 vec4_to_vec2(const float4& vec) {
    return make_float2(vec.x, vec.y);
}

__device__ __host__
float2 vec3_to_vec2(const float3& vec) {
    return make_float2(vec.x, vec.y);
}


// Operations
__device__ __host__
float length(const float2& a) {
    return sqrt(a.x * a.x + a.y * a.y);
}

__device__ __host__
float length(const float3& a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ __host__
float length(const float4& a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
}

__device__ __host__
float length2(const float2& a) {
    return a.x * a.x + a.y * a.y;
}

__device__ __host__
float length2(const float3& a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ __host__
float length2(const float4& a) {
    return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
}


template<typename T>
__device__ __host__
float distance(const T& a, const T& b) {
    T diff = a - b;

    return length(diff);
}

template<typename T>
__device__ __host__
float distance2(const T& a, const T& b) {
    T diff = a - b;

    return length2(diff);
}

template<typename T>
__device__ __host__
T normalise_to(const T& a, float b) {
    float len = length(a);
    if (len == 0) {
        return a;
    }
    
    return a * b / len;
}

template<typename T>
T normalize(const T& a) {
    float len = length(a);

    if (len == 0) {
        return a;
    }

    return a / len;
}