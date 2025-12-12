#define CV_CPU_SIMD_FILENAME "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv-4.10.0/modules/dnn/src/layers/cpu_kernels/fast_gemm_kernels.simd.hpp"
#define CV_CPU_DISPATCH_MODE AVX
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODE AVX2
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODE NEON
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODE LASX
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODES_ALL LASX, NEON, AVX2, AVX, BASELINE

#undef CV_CPU_SIMD_FILENAME
