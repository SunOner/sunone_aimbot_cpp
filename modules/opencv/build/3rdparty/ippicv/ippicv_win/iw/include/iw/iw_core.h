/*******************************************************************************
* Copyright 2016 Intel Corporation.
*
*
* This software and the related documents are Intel copyrighted materials, and your use of them is governed by
* the express license under which they were provided to you ('License'). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit this software or the related
* documents without Intel's prior written permission.
* This software and the related documents are provided as is, with no express or implied warranties, other than
* those that are expressly stated in the License.
*******************************************************************************/

#if !defined( __IPP_IW_CORE__ )
#define __IPP_IW_CORE__

#include "stddef.h" // NULL definition

#ifdef ICV_BASE
#include "ippicv.h"
#else
#include "ippcore.h"
#endif

#include "iw_version.h"

#ifdef __cplusplus
extern "C" {
#endif

// Intel IPP compatibility check
#if IPP_VERSION_COMPLEX < IW_MIN_COMPATIBLE_IPP_COMPLEX
#define IW_MACRO_TOS(A) #A
#define IW_VERSION_ERROR(MAJOR, MINOR, UPDATE) "warning: Unsupported Intel(R) IPP version. Minimal compatible version is " IW_MACRO_TOS(MAJOR)"." IW_MACRO_TOS(MINOR)"." IW_MACRO_TOS(UPDATE)
#ifdef _MSC_VER
#pragma message(IW_VERSION_ERROR(IW_MIN_COMPATIBLE_IPP_MAJOR, IW_MIN_COMPATIBLE_IPP_MINOR, IW_MIN_COMPATIBLE_IPP_UPDATE))
#else
#warning IW_VERSION_ERROR
#endif
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Base IW definitions
///////////////////////////////////////////////////////////////////////////// */

// Redefine calling convention for older Intel IPP packages
#ifndef IPP_STDCALL
#define IPP_STDCALL __STDCALL
#endif

// Default library linkage
#if !defined( IPP_NO_DEFAULT_LIB ) && !defined( IW_NO_DEFAULT_LIB )
  #if defined( _IPP_SEQUENTIAL_DYNAMIC )
    #pragma comment( lib, __FILE__ "/../../../../lib" IPP_LIB_SUBDIR_ARCH_SUFFIX "/ipp_iw" )
  #elif defined( _IPP_SEQUENTIAL_STATIC )
    #pragma comment( lib, __FILE__ "/../../../../lib" IPP_LIB_SUBDIR_ARCH_SUFFIX "/ipp_iw" )
  #elif defined( _IPP_PARALLEL_DYNAMIC )
    #pragma comment( lib, __FILE__ "/../../../../lib" IPP_LIB_SUBDIR_ARCH_SUFFIX "/ipp_iw" )
  #elif defined( _IPP_PARALLEL_STATIC )
    #pragma comment( lib, __FILE__ "/../../../../lib" IPP_LIB_SUBDIR_ARCH_SUFFIX "/ipp_iw" )
  #endif
#endif

// Common IW API declaration macro
#if defined IW_BUILD_DLL
#if defined _WIN32
#define IW_DECL(RET_TYPE) __declspec(dllexport) RET_TYPE IPP_STDCALL
#else
#define IW_DECL(RET_TYPE) RET_TYPE IPP_STDCALL
#endif
#else
#define IW_DECL(RET_TYPE) RET_TYPE IPP_STDCALL
#endif

#ifdef _MSC_VER
#define IW_INLINE __inline
#else
#define IW_INLINE inline
#endif

#define IwValueMin (-IPP_MAXABS_64F)
#define IwValueMax (IPP_MAXABS_64F)

typedef IppSizeL IwSize;

// Extended status declaration
#define iwStsErr                   -100000          // Extended errors offset
#define iwStsWrn                    100000          // Extended warnings offset

#define iwStsBorderNegSizeErr       iwStsErr-1      // Negative border size values are not allowed


// Transform direction enumerator
typedef enum _IwTransDirection
{
    iwTransForward = 0,
    iwTransInverse = 1
} IwTransDirection;

// CPUID helpers for ippSetCpuFeatures
#define iwCPU_SSE2          ippCPUID_MMX|ippCPUID_SSE|ippCPUID_SSE2
#define iwCPU_SSE3          iwCPU_SSE2|ippCPUID_SSE3
#define iwCPU_SSSE3         iwCPU_SSE3|ippCPUID_SSSE3
#define iwCPU_SSSE3_Atom    iwCPU_SSSE3|ippCPUID_MOVBE
#define iwCPU_SSE41         iwCPU_SSSE3|ippCPUID_SSE41
#define iwCPU_SSE42         iwCPU_SSE41|ippCPUID_SSE42|ippCPUID_AES
#define iwCPU_SSE42_Atom    iwCPU_SSE42|ippCPUID_AES|ippCPUID_CLMUL|ippCPUID_MOVBE|ippCPUID_SHA
#define iwCPU_AVX           iwCPU_SSE42|ippCPUID_AVX|ippCPUID_CLMUL|ippCPUID_RDRAND|ippCPUID_F16C
#if IPP_VERSION_COMPLEX >= 20170003
#define iwCPU_AVX2          iwCPU_AVX|ippCPUID_AVX2|ippCPUID_ADCOX|ippCPUID_RDSEED|ippCPUID_PREFETCHW|ippCPUID_MOVBE|ippCPUID_MPX
#else
#define iwCPU_AVX2          iwCPU_AVX|ippCPUID_AVX2|ippCPUID_ADCOX|ippCPUID_RDSEED|ippCPUID_PREFETCHW|ippCPUID_MOVBE
#endif
#define iwCPU_AVX512        iwCPU_AVX2|ippCPUID_AVX512F|ippCPUID_AVX512CD|\
                            ippCPUID_AVX512VL|ippCPUID_AVX512BW|ippCPUID_AVX512DQ
#define iwCPU_AVX512_KNL    iwCPU_SSE42_Atom|ippCPUID_AVX|ippCPUID_AVX2|\
                            ippCPUID_AVX512F|ippCPUID_AVX512CD|ippCPUID_AVX512PF|ippCPUID_AVX512ER


/* /////////////////////////////////////////////////////////////////////////////
//                   Base IW utility functions
///////////////////////////////////////////////////////////////////////////// */

// Convert IppDataType to actual size in bytes
// Returns:
//      Size of IppDataType in bytes
IW_DECL(int) iwTypeToSize(
    IppDataType type    // Data type
);

// Returns 1 if data type is of float type and 0 otherwise
// Returns:
//      Float flag
IW_DECL(int) iwTypeIsFloat(
    IppDataType type    // Data type
);

// Returns 1 if data type is of signed type and 0 otherwise
// Returns:
//      Signed flag
IW_DECL(int) iwTypeIsSigned(
    IppDataType type    // Data type
);

// Returns minimum possible value for specified data type
// Returns:
//      Minimum value
IW_DECL(double) iwTypeGetMin(
    IppDataType type    // Data type for min value
);

// Returns maximum possible value for specified data type
// Returns:
//      Maximum value
IW_DECL(double) iwTypeGetMax(
    IppDataType type    // Data type for max value
);

// Returns values range for specified data type
// Returns:
//      Range value
IW_DECL(double) iwTypeGetRange(
    IppDataType type    // Data type for range value
);

// Cast double value to input type with rounding and saturation
// Returns:
//      Rounded and saturated value
IW_DECL(double) iwValueSaturate(
    double      val,    // Input value
    IppDataType dstType // Data type for saturation range
);

// Converts relative value in range of [0,1] to the absolute value according to specified type
// Returns:
//      Absolute value
IW_DECL(double) iwValueRelToAbs(
    double      val,    // Relative value. From 0 to 1
    IppDataType type    // Data type for the absolute range
);

// Converts absolute value in range of the given type to the relative value in range [0,1]
// Returns:
//      Relative value
IW_DECL(double) iwValueAbsToRel(
    double      val,    // Absolute value
    IppDataType type    // Data type for the absolute range
);

// Returns relative weight disproportion of signed range, by adjusting to which middle of the signed range will be at 0 value
// Returns:
//      Relative correction value
IW_DECL(double) iwRangeWeightCorrector(
    IppDataType type    // Data type for the absolute range
);


/* /////////////////////////////////////////////////////////////////////////////
//                   IW with Threading Layer control
///////////////////////////////////////////////////////////////////////////// */

// This function sets number of threads for IW functions with parallel execution support
IW_DECL(void) iwSetThreadsNum(
    int threads     // Number of threads to use
);

// This function returns number of threads used by IW functions with parallel execution support
// Returns:
//      Number of threads or 0 if compiled without internal threading support
IW_DECL(int)  iwGetThreadsNum(void);

// This function returns initial number of threads used by IW functions with parallel execution support
// Returns:
//      Default number of threads or 0 if compiled without internal threading support
IW_DECL(int)  iwGetThreadsNumDefault(void);


/* /////////////////////////////////////////////////////////////////////////////
//                   IwAtomic - Atomic operations layer
///////////////////////////////////////////////////////////////////////////// */

// This function performs thread safe addition operation on integer variable.
// Returns:
//      Value of the variable before the operation
IW_DECL(int) iwAtomic_AddInt(int *pInt, int delta);

/* /////////////////////////////////////////////////////////////////////////////
//                   IwTls - TLS data storage interface
///////////////////////////////////////////////////////////////////////////// */
typedef void (IPP_STDCALL *IwTlsDestructor)(void*); // Pointer to destructor function for TLS object

// TLS abstraction layer structure
// This API can help with threading of IW functions by allowing easy platform-independent per-thread data storage and initialization
typedef struct _IwTls
{
    IwTlsDestructor m_desctuctor;  // Pointer to destruction function
    size_t          m_idx;         // Internal TLS index
    void           *m_pTlsStorage; // TLS object
} IwTls;

// This function initializes TLS structure, reserves TLS index and assign destruction function if necessary.
// Destruction function is required to properly deallocate user object for every thread.
// Returns:
//      ippStsErr                           internal TLS error
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwTls_Init(
    IwTls          *pTls,       // Pointer to IwTls structure
    IwTlsDestructor destructor  // Pointer to object destruction function
);

// Writes pointer to object into TLS storage for current thread. If data already exist IwTlsDestructor will be called first.
// Returns:
//      ippStsErr                           internal TLS error
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwTls_Set(
    IwTls *pTls, // Pointer to IwTls structure
    void  *pData // Pointer to user object
);

// Tries to get pointer to object from TLS storage for current thread.
// If no object has been yet created for current thread it returns NULL.
// Returns:
//      Pointer to stored data for current thread or NULL if no data was stored
IW_DECL(void*)     iwTls_Get(
    const IwTls *pTls // Pointer to IwTls structure
);

// Releases data for all threads, but not TLS object itself.
// Internal object data for different threads can be released here automatically only if IwTlsDestructor pointer was initialized
// Returns:
//      ippStsErr                           internal TLS error
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwTls_ReleaseData(
    IwTls *pTls // Pointer to IwTls structure
);

// Releases TLS object and all data associated with it for all threads.
// Internal object data for different threads can be released here automatically only if IwTlsDestructor pointer was initialized
// Returns:
//      ippStsErr                           internal TLS error
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwTls_Release(
    IwTls *pTls // Pointer to IwTls structure
);

/* /////////////////////////////////////////////////////////////////////////////
//                   IW version info
///////////////////////////////////////////////////////////////////////////// */

// IW version structure
typedef struct _IwVersion
{
    const IppLibraryVersion *m_pIppVersion;   // Pointer to version structure with version of linked Intel IPP library

    int         m_major;
    int         m_minor;
    int         m_update;
    const char* m_versionStr;
    int         m_bUserBuild;   // Manual build flag. Must be false for prebuilt library and true for custom user build
} IwVersion;

// Writes version information in IwVersion structure
IW_DECL(void) iwGetLibVersion(
    IwVersion *pVersion // Pointer to IwVersion structure
);

/* /////////////////////////////////////////////////////////////////////////////
//                   IW status
///////////////////////////////////////////////////////////////////////////// */

// Convert the library status code to a readable string
// This function supports extended status codes used in IW on top the of regular Intel IPP codes
// Returns:
//      Pointer to constant string describing the library status code
IW_DECL(const char*) iwGetStatusString(
    IppStatus status    // Library status code
);

#ifdef __cplusplus
}
#endif

#endif
