#ifndef KEYCODES_H
#define KEYCODES_H

#include <string>
#include <unordered_map>

class KeyCodes
{
public:
    static int getKeyCode(const std::string& key_name);
    static std::unordered_map<std::string, int> key_code_map;
};

#endif // KEYCODES_H