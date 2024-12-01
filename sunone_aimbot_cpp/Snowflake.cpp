#include "Snowflake.hpp"

Snowflake::vec3 gravity;

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
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

float GetRandomSize(float min, float max)
{
    float r = pow(RandomFloat(0.f, 1.f), 3);
    return Constrain(r * max, min, max);
}

Snowflake::Snowflake::Snowflake(float _minSize, float _maxSize, int _windowX, int _windowY, int _width, int _height, ImU32 _color)
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
    accelaretion = vec3();
    radius = GetRandomSize(minSize, maxSize);
}

void Snowflake::Snowflake::ApplyForce(vec3 force)
{
    vec3 f = force;
    f *= radius;
    accelaretion += f;
}

void Snowflake::Snowflake::Update()
{
    velocity += accelaretion;
    velocity.Limit(radius * 0.2f);

    pos += velocity;
    accelaretion *= 0.f;

    if (OffScreen())
        Randomize();
}

void Snowflake::Snowflake::Render()
{
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x, pos.y), radius, color);
}

bool Snowflake::Snowflake::OffScreen()
{
    return (pos.y > windowY + height + radius || pos.x < windowX - radius || pos.x > windowX + width + radius);
}

void Snowflake::Snowflake::Randomize()
{
    pos = vec3(RandomFloat(windowX, windowX + width), RandomFloat(windowY - 100.f, windowY - 10.f));
    velocity = vec3(0.f, 0.f);
    accelaretion = vec3();
    radius = GetRandomSize(minSize, maxSize);
}

bool Snowflake::Snowflake::operator==(const Snowflake& target)
{
    return pos == target.pos && velocity == target.velocity && accelaretion == target.accelaretion && radius == target.radius;
}

void Snowflake::CreateSnowFlakes(std::vector<Snowflake>& snow, uint64_t limit, float _minSize, float _maxSize, int _windowX, int _windowY, int _width, int _height, vec3 _gravity, ImU32 _color)
{
    gravity = _gravity;

    for (int i = 0; i < limit; i++)
        snow.push_back(Snowflake(_minSize, _maxSize, _windowX, _windowY, _width, _height, _color));
}

void Snowflake::Update(std::vector<Snowflake>& snow, vec3 mouse, vec3 windowPos)
{
    mouse.x -= windowPos.x;
    mouse.y -= windowPos.y;

    for (Snowflake& flake : snow)
    {
        float xOff = flake.pos.x / (flake.windowX + flake.width);
        float yOff = flake.pos.y / (flake.windowY + flake.height);
        float wx = Map(mouse.x - flake.windowX, 0, flake.width, -0.002f, 0.002f, true);
        vec3 wind = vec3(wx + (xOff * .002f), (yOff * .002f));
        wind *= .5f;

        flake.ApplyForce(gravity);
        flake.ApplyForce(wind);
        flake.Update();
        flake.Render();
    }
}

void Snowflake::ChangeWindowPos(std::vector<Snowflake>& snow, int _windowX, int _windowY)
{
    for (Snowflake& flake : snow)
    {
        flake.pos.x += _windowX - flake.windowX;
        flake.pos.y += _windowY - flake.windowY;
        flake.windowX = _windowX;
        flake.windowY = _windowY;
    }
}