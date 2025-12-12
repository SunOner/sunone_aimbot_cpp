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

#if !defined( __IPP_IWPP_CORE__ )
#define __IPP_IWPP_CORE__

#include "iw/iw_core.h"

#include <string>

// IW++ interface configuration switches
#ifndef IW_ENABLE_EXCEPTIONS
#define IW_ENABLE_EXCEPTIONS 1  // IW++ can return all errors by exceptions or by classic return codes.
                                // Note that some errors cannot be returned without exceptions and may be lost.
#endif


#if IW_ENABLE_EXCEPTIONS == 0
#define OWN_ERROR_THROW(IPP_STATUS)      return (IPP_STATUS);
#define OWN_ERROR_THROW_ONLY(IPP_STATUS) (void)(IPP_STATUS);
#else
#define OWN_ERROR_THROW(IPP_STATUS)      throw IwException(IPP_STATUS);
#define OWN_ERROR_THROW_ONLY(IPP_STATUS) OWN_ERROR_THROW(IPP_STATUS);
#endif
#define OWN_ERROR_CHECK_THROW_ONLY(IPP_STATUS)\
{\
    if((IPP_STATUS) < 0)\
    {\
        OWN_ERROR_THROW_ONLY(IPP_STATUS)\
    }\
}
#define OWN_ERROR_CHECK(IPP_STATUS)\
{\
    if((IPP_STATUS) < 0)\
    {\
        OWN_ERROR_THROW(IPP_STATUS)\
    }\
}

// Common IW++ API declaration macro
#define IW_DECL_CPP(RET_TYPE) IW_INLINE RET_TYPE IPP_STDCALL

// Base constructors set for auxiliary parameters
#define IW_BASE_PARAMS_CONSTRUCTORS(NAME, FUN)\
    NAME(const ::NAME *pParams)\
    {\
        if(pParams)\
            *((::NAME*)this) = *pParams;\
        else\
            FUN(this);\
    }\
    NAME(const ::NAME &params)\
    {\
        *((::NAME*)this) = params;\
    }\
    NAME(const ipp::IwDefault &)\
    {\
        FUN(this);\
    }

namespace ipp
{

/* /////////////////////////////////////////////////////////////////////////////
//                   Base IW++ definitions
///////////////////////////////////////////////////////////////////////////// */

using ::IppStatus;
using ::IwSize;

using ::Ipp8u;
using ::Ipp16u;
using ::Ipp32u;
using ::Ipp8s;
using ::Ipp16s;
using ::Ipp32s;
using ::Ipp32f;
using ::Ipp64s;
using ::Ipp64u;
using ::Ipp64f;
using ::Ipp16f;

using ::IppDataType;
using ::ipp1u;
using ::ipp8u;
using ::ipp8uc;
using ::ipp8s;
using ::ipp8sc;
using ::ipp16u;
using ::ipp16uc;
using ::ipp16s;
using ::ipp16sc;
using ::ipp32u;
using ::ipp32uc;
using ::ipp32s;
using ::ipp32sc;
using ::ipp32f;
using ::ipp32fc;
using ::ipp64u;
using ::ipp64uc;
using ::ipp64s;
using ::ipp64sc;
using ::ipp64f;
using ::ipp64fc;

// Class to initialize default objects
class IwDefault
{
public:
    IwDefault() {}
};

#if IW_ENABLE_EXCEPTIONS
// Stores an error code value for an exception thrown by the function
class IwException
{
public:
    // Constructor with status assignment
    IwException(
        IppStatus status    // IppStatus value
    )
    {
        m_status = status;
        m_string = iwGetStatusString(m_status);
    }

    // Default destructor
    ~IwException() {}

    // IwException to IppStatus cast operator
    inline operator IppStatus() const { return m_status;}

    IppStatus   m_status;   // Stored IppStatus value
    const char *m_string;   // Stored status string
};
#endif

// This class sets Intel IPP optimizations for the current region and restores previous optimizations at the region end
class IwSetCpuFeaturesRegion
{
public:
    // Default constructor. Saves current enabled CPU features.
    IwSetCpuFeaturesRegion()
    {
        m_stored = ::ippGetEnabledCpuFeatures();
    }

    // Saves current enabled CPU features and sets new features mask.
    IwSetCpuFeaturesRegion(Ipp64u featuresMask)
    {
        m_stored = ::ippGetEnabledCpuFeatures();
        Set(featuresMask);
    }

    // Sets new features mask for the region.
    IppStatus Set(Ipp64u featuresMask)
    {
        return ::ippSetCpuFeatures(featuresMask);
    }

    // Default destructor. Restores saved features mask.
    ~IwSetCpuFeaturesRegion()
    {
        ::ippSetCpuFeatures(m_stored);
    }

private:
    Ipp64u m_stored;
};

// Stores values for an array for array type casting
template<typename DST>
class IwValue
{
public:
    // Default constructor. Sets array to zero
    IwValue()
    {
        SetValue<Ipp8u>(0);
    }

    // Uniform template-based constructor. Sets channels to one value
    template<typename SRC>
    IwValue(SRC valUniform)
    {
        SetValue<SRC>(valUniform);
    }

    // 3 channels template-based constructor. Sets channels to individual values
    template<typename SRC>
    IwValue(SRC valC1, SRC valC2, SRC valC3)
    {
        SetValue<SRC>(valC1, valC2, valC3);
    }

    // 4 channels template-based constructor. Sets channels to individual values
    template<typename SRC>
    IwValue(SRC valC1, SRC valC2, SRC valC3, SRC valC4)
    {
        SetValue<SRC>(valC1, valC2, valC3, valC4);
    }

    // Buffer template-based constructor. Sets values from a buffer of specific type
    template<typename SRC>
    IwValue(SRC *pBuffer, int channels)
    {
        SetValue(pBuffer, channels);
    }

    // Buffer parameter-based constructor. Sets values from a buffer of specific type
    IwValue(void *pBuffer, IppDataType type, int channels)
    {
        SetValue(pBuffer, type, channels);
    }

    // Uniform template setter. Sets channels to one value
    template<typename SRC>
    void SetValue(SRC valUniform)
    {
        m_values = 1;
        m_val[0] = m_val[1] = m_val[2] = m_val[3] = (DST)valUniform;
    }

    // 3 channels template setter. Sets channels to individual values
    template<typename SRC>
    void SetValue(SRC valC1, SRC valC2, SRC valC3)
    {
        m_values = 3;
        m_val[0] = (DST)valC1, m_val[1] = (DST)valC2, m_val[2] = (DST)valC3;
    }

    // 4 channels template setter. Sets channels to individual values
    template<typename SRC>
    void SetValue(SRC valC1, SRC valC2, SRC valC3, SRC valC4)
    {
        m_values = 4;
        m_val[0] = (DST)valC1, m_val[1] = (DST)valC2, m_val[2] = (DST)valC3, m_val[3] = (DST)valC4;
    }

    // Buffer template-based setter. Sets values from a buffer of specific type
    template<typename SRC>
    void SetValue(SRC *pBuffer, int channels)
    {
        if(!pBuffer)
            OWN_ERROR_THROW_ONLY(ippStsNullPtrErr);
        if(channels > 4 || channels < 1)
            OWN_ERROR_THROW_ONLY(ippStsNumChannelsErr);
        m_values = channels;

        for(int i = 0; i < channels; i++)
            m_val[i] = (DST)(pBuffer[i]);
    }

    // Buffer parameter-based setter. Sets values from a buffer of specific type
    void SetValue(void *pBuffer, IppDataType type, int channels)
    {
        switch(type)
        {
            case ipp8u:
                SetValue((Ipp8u*)pBuffer, channels);  break;
            case ipp8s:
                SetValue((Ipp8s*)pBuffer, channels);  break;
            case ipp16u:
                SetValue((Ipp16u*)pBuffer, channels); break;
            case ipp16s:
                SetValue((Ipp16s*)pBuffer, channels); break;
            case ipp32u:
                SetValue((Ipp32u*)pBuffer, channels); break;
            case ipp32s:
                SetValue((Ipp32s*)pBuffer, channels); break;
            case ipp32f:
                SetValue((Ipp32f*)pBuffer, channels); break;
            case ipp64u:
                SetValue((Ipp64u*)pBuffer, channels); break;
            case ipp64s:
                SetValue((Ipp64s*)pBuffer, channels); break;
            case ipp64f:
                SetValue((Ipp64f*)pBuffer, channels); break;
            default:
                OWN_ERROR_THROW_ONLY(ippStsDataTypeErr);
        }
    }

    // Returns number of initialized values
    int ValuesNum() const { return m_values; }

    // IwValue to Ipp64f cast operator
    inline operator       DST  () const { return ((DST*)m_val)[0];}

    // IwValue to Ipp64f* cast operator
    inline operator       DST* () const { return (DST*)m_val;}

    // IwValue to const Ipp64f* cast operator
    inline operator const DST* () const { return (const DST*)m_val;}

    // Array subscript operator
    inline DST& operator[](int channel)             { return (channel>=m_values)?m_val[m_values-1]:m_val[channel]; }
    inline const DST& operator[](int channel) const { return (channel>=m_values)?m_val[m_values-1]:m_val[channel]; }

    // Compares values
    bool operator==(const IwValue& rhs) const
    {
        if((*this)[0] == rhs[0] && (*this)[1] == rhs[1] && (*this)[2] == rhs[2] && (*this)[3] == rhs[3])
            return true;
        else
            return false;
    }
    bool operator!=(const IwValue& rhs) const
    {
        return !(*this==rhs);
    }

private:
    int    m_values;  // Number of initialized values
    DST    m_val[4];  // reserve 4 channels
};

typedef IwValue<Ipp64f> IwValueFloat;
typedef IwValue<Ipp32s> IwValueInt;

// Convert IppDataType to actual size in bytes
// Returns:
//      Size of IppDataType in bytes
IW_DECL_CPP(int) iwTypeToSize(
    IppDataType type    // Data type
)
{
    return ::iwTypeToSize(type);
}

// Returns 1 if data type is of float type and 0 otherwise
// Returns:
//      Absolute value
IW_DECL_CPP(int) iwTypeIsFloat(
    IppDataType type    // Data type
)
{
    return ::iwTypeIsFloat(type);
}

// Returns minimum possible value for specified data type
// Returns:
//      Minimum value
IW_DECL_CPP(double) iwTypeGetMin(
    IppDataType type    // Data type for min value
)
{
    return ::iwTypeGetMin(type);
}

// Returns maximum possible value for specified data type
// Returns:
//      Maximum value
IW_DECL_CPP(double) iwTypeGetMax(
    IppDataType type    // Data type for max value
)
{
    return ::iwTypeGetMax(type);
}

// Returns values range for specified data type
// Returns:
//      Range value
IW_DECL_CPP(double) iwTypeGetRange(
    IppDataType type    // Data type for range value
)
{
    return ::iwTypeGetRange(type);
}

// Cast double value to input type with rounding and saturation
// Returns:
//      Rounded and saturated value
IW_DECL_CPP(double) iwValueSaturate(
    double      val,    // Input value
    IppDataType dstType // Data type for saturation range
)
{
    return ::iwValueSaturate(val, dstType);
}

// Converts relative value in range of [0,1] to the absolute value according to specified type
// Returns:
//      Absolute value
IW_DECL_CPP(double) iwValueRelToAbs(
    double      val,    // Relative value. From 0 to 1
    IppDataType type    // Data type for the absolute range
)
{
    return ::iwValueRelToAbs(val, type);
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IW with Threading Layer control
///////////////////////////////////////////////////////////////////////////// */

// This function sets number of threads for IW functions with parallel execution support
IW_DECL_CPP(void) iwSetThreadsNum(
    int threads     // Number of threads to use
)
{
    ::iwSetThreadsNum(threads);
}

// This function returns number of threads used by IW functions with parallel execution support
// Returns:
//      Number of threads or 0 if compiled without internal threading support
IW_DECL_CPP(int)  iwGetThreadsNum()
{
    return ::iwGetThreadsNum();
}

// This function returns initial number of threads used by IW functions with parallel execution support
// Returns:
//      Default number of threads or 0 if compiled without internal threading support
IW_DECL_CPP(int)  iwGetThreadsNumDefault()
{
    return ::iwGetThreadsNumDefault();
}


/* /////////////////////////////////////////////////////////////////////////////
//                   IwTls - TLS data storage interface
///////////////////////////////////////////////////////////////////////////// */

// Template-based TLS abstraction layer class.
// This is an extension of C IwTls structure with automatic objects destruction
template<class TYPE>
class IwTls: private ::IwTls
{
public:
    // Default constructor
    IwTls()
    {
        IppStatus status = ::iwTls_Init(this, (IwTlsDestructor)(IwTls::TypeDestructor));
        OWN_ERROR_CHECK_THROW_ONLY(status);
    }

    // Default destructor
    ~IwTls()
    {
        ::iwTls_Release(this);
    }

    // Allocates object for current thread and returns pointer to it
    TYPE* Create()
    {
        TYPE *pData = new TYPE;
        if(!pData)
            return NULL;
        IppStatus status = ::iwTls_Set(this, pData);
        if(status < 0)
        {
            delete pData;
            OWN_ERROR_CHECK_THROW_ONLY(status);
            return NULL;
        }
        return pData;
    }

    // Releases object for current thread
    void Release()
    {
        IppStatus status = ::iwTls_Set(this, NULL);
        if(status < 0)
            OWN_ERROR_CHECK_THROW_ONLY(status);
    }

    // Releases objects for all threads
    void ReleaseAll()
    {
        IppStatus status = ::iwTls_ReleaseData(this);
        if(status < 0)
            OWN_ERROR_CHECK_THROW_ONLY(status);
    }

    // Returns pointer to object for current thread
    TYPE* Get() const
    {
        return (TYPE*)::iwTls_Get(this);
    }

private:
    // Object destructor
    static void IPP_STDCALL TypeDestructor(void *pData)
    {
        if(pData)
            delete ((TYPE*)pData);
    }
};

/* /////////////////////////////////////////////////////////////////////////////
//                   IW version info
///////////////////////////////////////////////////////////////////////////// */

class IppVersion: private ::IwVersion
{
public:
    IppVersion()
    {
        ::iwGetLibVersion(this);
        if(!this->m_pIppVersion)
            OWN_ERROR_THROW_ONLY(ippStsNullPtrErr);
    }

    int getMajor()
    {
        return this->m_pIppVersion->major;
    }

    int getMinor()
    {
        return this->m_pIppVersion->minor;
    }

    int getUpdate()
    {
        return this->m_pIppVersion->majorBuild;
    }

    int getRevision()
    {
        return this->m_pIppVersion->build;
    }

    std::string getString()
    {
        return this->m_pIppVersion->Version;
    }

    std::string getLibraryName()
    {
        return this->m_pIppVersion->Name;
    }

    std::string getTargetCpu()
    {
        return this->m_pIppVersion->targetCpu;
    }

    std::string getBuildDate()
    {
        return this->m_pIppVersion->BuildDate;
    }

    std::string getInfoString()
    {
        return getLibraryName() + " " + getString() + ", " + getBuildDate();
    }
};

class IwVersion: private ::IwVersion
{
public:
    IwVersion()
    {
        ::iwGetLibVersion(this);
        if(!this->m_pIppVersion)
            OWN_ERROR_THROW_ONLY(ippStsNullPtrErr);
    }

    int getMajor()
    {
        return this->m_major;
    }

    int getMinor()
    {
        return this->m_minor;
    }

    int getUpdate()
    {
        return this->m_update;
    }

    std::string getString()
    {
        return this->m_versionStr;
    }

    std::string getInfoString()
    {
        if(isReleaseBuild())
            return getString() + ", release build";
        else
            return getString() + ", user build";
    }

    bool isReleaseBuild()
    {
        return !this->m_bUserBuild;
    }
};

}

#endif
