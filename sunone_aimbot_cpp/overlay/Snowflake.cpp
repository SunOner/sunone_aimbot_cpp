#include "Snowflake.hpp"
#include <chrono>
#include <GLFW/glfw3.h>
#include <cmath>

namespace Snowflake
{
    vec3 gravity;

    float min(float a, float b)
    {
        return (((a) < (b)) ? (a) : (b));
    }

    float max(float a, float b)
    {
        return (((a) > (b)) ? (a) : (b));
    }

    float Constrain(float n, float low, float high)
    {
        return max(min(n, high), low);
    }

    float Map(float n, float start1, float stop1, float start2, float stop2, bool withinBounds = false)
    {
        const float newVal = (n - start1) / (stop1 - start1) * (stop2 - start2) + start2;
        if (!withinBounds)
            return newVal;

        if (start2 < stop2)
            return Constrain(newVal, start2, stop2);
        else
            return Constrain(newVal, stop2, start2);
    }

    float RandomFloat(float a, float b)
    {
        std::uniform_real_distribution<float> distribution(a, b);
        return distribution(generator);
    }

    float GetRandomSize(float min, float max)
    {
        std::normal_distribution<float> distribution((min + max) / 2, (max - min) / 4);
        float size = distribution(generator);
        return Constrain(size, min, max);
    }

    float PerlinNoise(float x)
    {
        int xi = static_cast<int>(x) & 255;
        float xf = x - static_cast<int>(x);
        float u = xf * xf * (3 - 2 * xf);

        float rand1 = RandomFloat(-1.f, 1.f);
        float rand2 = RandomFloat(-1.f, 1.f);

        return (1 - u) * rand1 + u * rand2;
    }

    Snowflake::Snowflake(float _minSize, float _maxSize, int _windowX, int _windowY, int _width, int _height, ImU32 _color)
    {
        minSize = _minSize;
        maxSize = _maxSize;
        windowX = _windowX;
        windowY = _windowY;
        width = _width;
        height = _height;
        color = _color;
        pos = vec3(RandomFloat(windowX, windowX + width), RandomFloat(windowY - 100.f, windowY - 10.f));
        velocity = vec3(0.f, 0.f);
        acceleration = vec3();
        radius = GetRandomSize(minSize, maxSize);
        timeOffset = RandomFloat(0.f, 1000.f);
        opacity = 0.0f;

        angle = RandomFloat(0.f, 2.f * 3.1415926535f);
        dir = RandomFloat(-1.f, 1.f);
    }

    void Snowflake::ApplyForce(vec3 force)
    {
        float mass = Map(radius, minSize, maxSize, 0.5f, 2.0f);
        vec3 f = force;
        f /= mass;
        acceleration += f;
    }

    void Snowflake::Update(float deltaTime)
    {
        float time = glfwGetTime() + timeOffset;
        float windStrength = Map(radius, minSize, maxSize, 0.1f, 0.01f);
        float turbulenceX = PerlinNoise(time * 0.1f + pos.y * 0.01f) * windStrength * deltaTime * 100.f;
        float turbulenceY = PerlinNoise(time * 0.1f + pos.x * 0.01f) * windStrength * deltaTime * 100.f;

        vec3 turbulence(turbulenceX, turbulenceY);

        ApplyForce(turbulence);

        angle += dir * velocity.Mag() / 500.f;

        velocity += acceleration;
        velocity.Limit(radius * 0.5f);
        pos += velocity;
        acceleration *= 0.f;

        float fadeSpeed = 0.5f;
        if (pos.y < windowY + height / 2)
        {
            opacity += fadeSpeed * deltaTime;
        }
        else
        {
            opacity -= fadeSpeed * deltaTime;
        }
        opacity = Constrain(opacity, 0.0f, 1.0f);

        if (OffScreen())
            Randomize();
    }

    void Snowflake::Render()
    {
        ImU32 renderColor = ImGui::GetColorU32(ImVec4(
            ((color >> IM_COL32_R_SHIFT) & 0xFF) / 255.0f,
            ((color >> IM_COL32_G_SHIFT) & 0xFF) / 255.0f,
            ((color >> IM_COL32_B_SHIFT) & 0xFF) / 255.0f,
            opacity));

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->AddCircleFilled(ImVec2(pos.x, pos.y), radius, renderColor);
    }

    bool Snowflake::OffScreen()
    {
        return (pos.y > windowY + height + radius || pos.x < windowX - radius || pos.x > windowX + width + radius);
    }

    void Snowflake::Randomize()
    {
        pos = vec3(RandomFloat(windowX - 100.f, windowX + width + 100.f), RandomFloat(windowY - 200.f, windowY - 10.f));
        velocity = vec3(0.f, 0.f);
        acceleration = vec3();
        radius = GetRandomSize(minSize, maxSize);

        angle = RandomFloat(0.f, 2.f * 3.1415926535f);
        dir = RandomFloat(-1.f, 1.f);
    }

    bool Snowflake::operator==(const Snowflake& target)
    {
        return pos == target.pos && velocity == target.velocity && acceleration == target.acceleration && radius == target.radius;
    }

    void CreateSnowFlakes(std::vector<Snowflake>& snow, uint64_t limit, float _minSize, float _maxSize, int _windowX, int _windowY, int _width, int _height, vec3 _gravity, ImU32 _color)
    {
        gravity = _gravity;

        for (uint64_t i = 0; i < limit; i++)
            snow.push_back(Snowflake(_minSize, _maxSize, _windowX, _windowY, _width, _height, _color));
    }

    void Update(std::vector<Snowflake>& snow, vec3 windowPos, float deltaTime)
    {
        for (Snowflake& flake : snow)
        {
            flake.ApplyForce(gravity);
            flake.Update(deltaTime);
            flake.Render();
        }
    }

    void ChangeWindowPos(std::vector<Snowflake>& snow, int _windowX, int _windowY)
    {
        for (Snowflake& flake : snow)
        {
            flake.pos.x += _windowX - flake.windowX;
            flake.pos.y += _windowY - flake.windowY;
            flake.windowX = _windowX;
            flake.windowY = _windowY;
        }
    }
}