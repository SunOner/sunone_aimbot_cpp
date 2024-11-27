#ifndef OTHER_TOOLS_H
#define OTHER_TOOLS_H

#include <string>
#include <d3d11.h>

static inline bool is_base64(unsigned char c);
std::vector<unsigned char> Base64Decode(const std::string& encoded_string);
bool fileExists(const std::string& path);
std::string replace_extension(const std::string& filename, const std::string& new_extension);
std::vector<std::string> getEngineFiles();
std::vector<std::string> getModelFiles();
std::vector<std::string> getOnnxFiles();
std::vector<std::string>::difference_type getModelIndex(std::vector<std::string> engine_models);
int getImageSizeIndex(int engine_image_size, const int* model_sizes, int model_sizes_count);
void HideConsole();
void ShowConsole();
bool IsConsoleVisible();
std::string intToString(int value);
bool LoadTextureFromFile(const char* filename, ID3D11Device* device, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height);
bool LoadTextureFromMemory(const std::string& imageBase64, ID3D11Device* device, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height);
std::string get_ghub_version();
std::string get_environment_vars();
std::string get_tensorrt_path();
int get_active_monitors();
HMONITOR GetMonitorHandleByIndex(int monitorIndex);
#endif // OTHER_TOOLS_H