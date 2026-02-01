#ifndef OTHER_TOOLS_H
#define OTHER_TOOLS_H

#include <string>
#include <vector>
#include <d3d11.h>

static inline bool is_base64(unsigned char c);
std::vector<unsigned char> Base64Decode(const std::string& encoded_string);
bool fileExists(const std::string& path);
std::string replace_extension(const std::string& filename, const std::string& new_extension);

std::vector<std::string> getEngineFiles();
std::vector<std::string> getModelFiles();
std::vector<std::string> getOnnxFiles();
std::vector<std::string>::difference_type getModelIndex(const std::vector<std::string>& engine_models);

std::string intToString(int value);
bool LoadTextureFromFile(const char* filename, ID3D11Device* device, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height);
bool LoadTextureFromMemory(const std::string& imageBase64, ID3D11Device* device, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height);

std::string get_ghub_version();
int get_active_monitors();
HMONITOR GetMonitorHandleByIndex(int monitorIndex);
void SetRandomConsoleTitle();
bool IsValidImageFile(const std::wstring& wpath, UINT& outW, UINT& outH, std::string& outErr);

std::vector<std::string> getAvailableModels();

void welcome_message();
bool checkwin1903();
std::string WideToUtf8(const std::wstring& ws);
#endif // OTHER_TOOLS_H
