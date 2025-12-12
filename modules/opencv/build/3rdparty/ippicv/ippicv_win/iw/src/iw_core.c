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

#include "iw_own.h"
#include "iw/iw_image.h"

#if defined _WIN32
    #include <malloc.h>
    #include <intrin.h>
#else
    #ifdef _OPENMP
        #if (defined __GNUC__) && !(defined __clang__)
            #define GCC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
            #if (GCC_VERSION >= 40700)
                #define OWN_ALLOW_OMP_ATOMICS
            #endif
            #undef GCC_VERSION
        #else
            #define OWN_ALLOW_OMP_ATOMICS
        #endif
    #endif

    #ifdef OWN_ALLOW_OMP_ATOMICS
        #include <omp.h> // Use OMP atomics
    #else
        #if (defined __clang__ && defined __has_include)
            #if !__has_include(<stdatomic.h>)
                #ifndef __STDC_NO_ATOMICS__
                    #define __STDC_NO_ATOMICS__
                #endif
            #endif
        #elif (defined __GNUC__)
            #define GCC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
            #if (GCC_VERSION < 40900)
                #ifndef __STDC_NO_ATOMICS__
                    #define __STDC_NO_ATOMICS__
                #endif
            #endif
            #undef GCC_VERSION
        #endif

        #if !defined __STDC_NO_ATOMICS__
            #include <stdatomic.h>
            #ifndef __ATOMIC_ACQ_REL
                #define __ATOMIC_ACQ_REL 4
            #endif
        #else
            #pragma message("Atomic operations are not supported by this compiler. Some features my not be thread-safe.")
        #endif
    #endif
        #ifndef __APPLE__
            #include <malloc.h>
        #endif
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   IW DLL entry points
///////////////////////////////////////////////////////////////////////////// */
#ifdef IW_BUILD_DLL
#if defined _WIN32
#include <Windows.h>
int WINAPI DllMain( HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved )
{
   switch( fdwReason )
   {
    case DLL_PROCESS_ATTACH:    break;
    case DLL_THREAD_ATTACH:     break;
    case DLL_THREAD_DETACH:     break;
    case DLL_PROCESS_DETACH:    break;
    default: break;
   }
   return 1;
   UNREFERENCED_PARAMETER(hinstDLL);
   UNREFERENCED_PARAMETER(lpvReserved);
}
#elif defined __unix__
int _init(void)
{
    return 1;
}

void _fini(void)
{
}
#elif defined __APPLE__
__attribute__((constructor)) void initializer( void )
{
    static int initialized = 0;
    if(!initialized)
    {
        initialized = 1;
    }

    return;
}

__attribute__((destructor)) void destructor()
{
}
#endif
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Base IW definitions
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(int) iwTypeToSize(IppDataType dataType)
{
    switch(dataType)
    {
    case ipp8u:
    case ipp8s:
        return 1;
    case ipp8uc:
    case ipp8sc:
    case ipp16u:
    case ipp16s:
        return 2;
    case ipp16uc:
    case ipp16sc:
    case ipp32u:
    case ipp32s:
    case ipp32f:
        return 4;
    case ipp32uc:
    case ipp32sc:
    case ipp32fc:
    case ipp64u:
    case ipp64s:
    case ipp64f:
        return 8;
    case ipp64uc:
    case ipp64sc:
    case ipp64fc:
        return 16;
    default:
        return 0;
    }
}

IW_DECL(double) iwTypeGetMin(IppDataType type)
{
    switch(type)
    {
    case ipp8u:  return IPP_MIN_8U;
    case ipp8s:  return IPP_MIN_8S;
    case ipp16u: return IPP_MIN_16U;
    case ipp16s: return IPP_MIN_16S;
    case ipp32u: return IPP_MIN_32U;
    case ipp32s: return IPP_MIN_32S;
    case ipp32f: return -IPP_MAXABS_32F;
    case ipp64f: return -IPP_MAXABS_64F;
    default:     return 0;
    }
}

IW_DECL(double) iwTypeGetMax(IppDataType type)
{
    switch(type)
    {
    case ipp8u:  return IPP_MAX_8U;
    case ipp8s:  return IPP_MAX_8S;
    case ipp16u: return IPP_MAX_16U;
    case ipp16s: return IPP_MAX_16S;
    case ipp32u: return IPP_MAX_32U;
    case ipp32s: return IPP_MAX_32S;
    case ipp32f: return IPP_MAXABS_32F;
    case ipp64f: return IPP_MAXABS_64F;
    default:     return 0;
    }
}

IW_DECL(double) iwTypeGetRange(IppDataType type)
{
    switch(type)
    {
    case ipp8u:  return ((double)IPP_MAX_8U  - IPP_MIN_8U);
    case ipp8s:  return ((double)IPP_MAX_8S  - IPP_MIN_8S);
    case ipp16u: return ((double)IPP_MAX_16U - IPP_MIN_16U);
    case ipp16s: return ((double)IPP_MAX_16S - IPP_MIN_16S);
    case ipp32u: return ((double)IPP_MAX_32U - IPP_MIN_32U);
    case ipp32s: return ((double)IPP_MAX_32S - IPP_MIN_32S);
    default:     return 0;
    }
}

IW_DECL(int) iwTypeIsFloat(IppDataType type)
{
    return (type == ipp64f || type == ipp64fc || type == ipp32f || type == ipp32fc)?1:0;
}

IW_DECL(int) iwTypeIsSigned(IppDataType type)
{
    return (type == ipp64f || type == ipp64fc || type == ipp64s || type == ipp64sc ||
        type == ipp32f || type == ipp32fc || type == ipp32s || type == ipp32sc ||
        type == ipp16s || type == ipp16sc || type == ipp8s || type == ipp8sc)?1:0;
}

IW_DECL(double) iwValueSaturate(double val, IppDataType dstType)
{
    switch(dstType)
    {
    case ipp8u:  return (double)ownCast_64f8u(val);
    case ipp8s:  return (double)ownCast_64f8s(val);
    case ipp16u: return (double)ownCast_64f16u(val);
    case ipp16s: return (double)ownCast_64f16s(val);
    case ipp32u: return (double)ownCast_64f32u(val);
    case ipp32s: return (double)ownCast_64f32s(val);
    default:     return val;
    }
}

IW_DECL(double) iwValueRelToAbs(double val, IppDataType type)
{
    if(iwTypeIsFloat(type))
        return val;
    else
    {
        double min = iwTypeGetMin(type);
        double max = iwTypeGetMax(type);
        return (max - min)*val + min;
    }
}

IW_DECL(double) iwValueAbsToRel(double val, IppDataType type)
{
    if(iwTypeIsFloat(type))
        return val;
    else
    {
        double min = iwTypeGetMin(type);
        double max = iwTypeGetMax(type);
        return (val - min)/(max - min);
    }
}

IW_DECL(double) iwRangeWeightCorrector(IppDataType type)
{
    if(iwTypeIsSigned(type) && !iwTypeIsFloat(type))
    {
        double min   = iwTypeGetMin(type);
        double max   = iwTypeGetMax(type);
        double range = iwTypeGetRange(type);
        if(range)
            return (-min-max)/range;
        else
            return 0;
    }
    return 0;
}


/* /////////////////////////////////////////////////////////////////////////////
//                   IwAtomic - Atomic operations layer
///////////////////////////////////////////////////////////////////////////// */

IW_DECL(int) iwAtomic_AddInt(int *pInt, int delta)
{
#if defined _WIN32
    return _InterlockedExchangeAdd((long volatile*)pInt, delta);
#else
#ifdef OWN_ALLOW_OMP_ATOMICS
    int ret;
    #pragma omp atomic capture
    {
        ret = *pInt;
        *pInt += delta;
    }
    return ret;
#else
#if defined __APPLE__ && !defined __STDC_NO_ATOMICS__
    return __atomic_fetch_add(pInt, delta, __ATOMIC_ACQ_REL);
#elif defined __GNUC__ && !defined __STDC_NO_ATOMICS__
    return __atomic_fetch_add(pInt, delta, __ATOMIC_ACQ_REL);
#else
    int ret = *pInt;
    *pInt  += delta;
    return ret;
#endif
#endif
#endif
}


/* /////////////////////////////////////////////////////////////////////////////
//                   IW version info
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(void)  iwGetLibVersion(IwVersion *pVersion)
{
    if(!pVersion)
        return;

    pVersion->m_major       = IW_VERSION_MAJOR;
    pVersion->m_minor       = IW_VERSION_MINOR;
    pVersion->m_update      = IW_VERSION_UPDATE;
    pVersion->m_versionStr  = IW_VERSION_STR;
    pVersion->m_pIppVersion = ippiGetLibVersion();
#ifdef IW_PREBUILT
    pVersion->m_bUserBuild = 0;
#else
    pVersion->m_bUserBuild = 1;
#endif
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IW status
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(const char*) iwGetStatusString(IppStatus status)
{
#ifdef ICV_BASE
    (void)status;
    return "Status messages are not supported";
#else
    if(status <= iwStsErr)
        return ippGetStatusString(status);
    else if(status >= iwStsWrn)
        return ippGetStatusString(status);
    else
        return ippGetStatusString(status);
#endif
}
