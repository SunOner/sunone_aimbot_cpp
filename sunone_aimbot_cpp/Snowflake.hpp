#pragma once
#include <cmath>
#include <vector>
#include "imgui.h"

namespace Snowflake
{
    struct vec3
    {
        float x, y, z;

        vec3()
        {
            x = 0.f;
            y = 0.f;
            z = 0.f;
        }

        vec3(float tX, float tY, float tZ)
        {
            x = tX;
            y = tY;
            z = tZ;
        }

        vec3(float tX, float tY)
        {
            x = tX;
            y = tY;
            z = 0.f;
        }

        vec3 operator+(const vec3& target)
        {
            return vec3(x + target.x, y + target.y, z + target.z);
        }

        vec3& operator+=(const vec3& target)
        {
            x += target.x;
            y += target.y;
            z += target.z;

            return *this;
        }

        vec3& operator*=(const float& target)
        {
            x *= target;
            y *= target;
            z *= target;

            return *this;
        }

        bool operator==(const vec3& target)
        {
            return x == target.x && y == target.y && z == target.z;
        }

        vec3& operator/=(const float& target)
        {
            x /= target;
            y /= target;
            z /= target;

            return *this;
        }

        float MagSq()
        {
            return x * x + y * y + z * z;
        }

        float Mag()
        {
            return sqrt(MagSq());
        }

        vec3 Limit(float max)
        {
            float mSq = MagSq();
            if (mSq > max * max)
            {
                *this /= sqrt(mSq);
                *this *= max;
            }
            return *this;
        }

        vec3 Normalize()
        {
            float len = Mag();
            if (len != 0)
                *this *= (1 / len);
            return *this;
        }
    };

    class Snowflake
    {
    private:
        vec3 velocity;
        vec3 accelaretion;
        float radius;
        ImU32 color;
        float minSize;
        float maxSize;

    public:
        int width;
        int height;
        int windowX;
        int windowY;
        vec3 pos;

        Snowflake(float _minSize, float _maxSize, int _windowX, int _windowY, int _width, int _height, ImU32 _color);
        void ApplyForce(vec3 force);
        void Update();
        void Render();
        bool OffScreen();
        void Randomize();
        bool operator==(const Snowflake& target);
    };

    void CreateSnowFlakes(std::vector<Snowflake>& snow, uint64_t limit, float _minSize, float _maxSize, int _windowX, int _windowY, int _width, int _height, vec3 _gravity, ImU32 _color);
    void Update(std::vector<Snowflake>& snow, vec3 mousePos, vec3 windowPos);
    void ChangeWindowPos(std::vector<Snowflake>& snow, int _windowX, int _windowY);
}