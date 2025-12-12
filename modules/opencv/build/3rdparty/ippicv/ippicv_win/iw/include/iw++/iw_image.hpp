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

#if !defined( __IPP_IWPP_IMAGE__ )
#define __IPP_IWPP_IMAGE__

#include "iw++/iw_core.hpp"
#include "iw/iw_image_op.h"

namespace ipp
{

/* /////////////////////////////////////////////////////////////////////////////
//                   Image IW++ definitions
///////////////////////////////////////////////////////////////////////////// */

using ::IppiAxis;
using ::ippAxsHorizontal;
using ::ippAxsVertical;
using ::ippAxsBoth;
using ::ippAxs45;
using ::ippAxs135;

using ::IppiMaskSize;
using ::ippMskSize3x3;
using ::ippMskSize5x5;

using ::ippBorderRepl;
using ::ippBorderMirror;
using ::ippBorderDefault;
using ::ippBorderConst;
using ::ippBorderTransp;
using ::ippBorderInMemTop;
using ::ippBorderInMemBottom;
using ::ippBorderInMemLeft;
using ::ippBorderInMemRight;
using ::ippBorderInMem;
#if IPP_VERSION_COMPLEX >= 20170002
using ::ippBorderFirstStageInMem;
#endif
#if IPP_VERSION_COMPLEX >= 20180000
using ::ippBorderFirstStageInMemTop;
using ::ippBorderFirstStageInMemBottom;
using ::ippBorderFirstStageInMemLeft;
using ::ippBorderFirstStageInMemRight;
#endif

// Stores the width and height of a rectangle. Extends IwiSize structure
class IwiSize: public ::IwiSize
{
public:
    // Default constructor. Sets values to zero
    IwiSize()
    {
        Set(0, 0);
    }

    // One value template constructor. Sets size to same value. Useful for simple initialization, e.g.: size = 0
    template<typename T>
    IwiSize(
        T size // Size of square rectangle
    )
    {
        Set((IwSize)size, (IwSize)size);
    }

    // Constructor. Sets size to specified values
    IwiSize(
        IwSize _width, // Width of rectangle
        IwSize _height // Height of rectangle
    )
    {
        Set(_width, _height);
    }

    // Constructor from IppiSize structure
    IwiSize(
        IppiSize size // IppiSize structure
    )
    {
        Set(size.width, size.height);
    }

    // Constructor from C IwiSize structure
    IwiSize(
        ::IwiSize size // C IwiSize structure
    )
    {
        Set(size.width, size.height);
    }

    // Constructor from IppiRect structure
    IwiSize(
        IppiRect rect // IppiRect structure
    )
    {
        Set(rect.width, rect.height);
    }

    // Constructor from IwiRoi structure
    IwiSize(
        IwiRoi rect // IwiRoi structure
    )
    {
        Set(rect.width, rect.height);
    }

    // Sets size to specified values
    void Set(
        IwSize _width, // Width of rectangle
        IwSize _height // Height of rectangle
    )
    {
        width  = _width;
        height = _height;
    }

    // Retrns size of area covered by IwiSize
    inline IwSize Area() const
    {
        return IPP_ABS(this->width*this->height);
    }

    // IwiSize to IppiSize cast operator
    inline operator IppiSize()  const { IppiSize size = {(int)width, (int)height}; return size; }
};

// Stores the geometric position of a point. Extends IppiPoint structure
class IwiPoint: public ::IwiPoint
{
public:
    // Default constructor. Sets values to zero
    IwiPoint()
    {
        Set(0, 0);
    }

    // One value template constructor. Sets position to same value. Useful for simple initialization, e.g.: point = 0
    template<typename T>
    IwiPoint(
        T point // Position of point
    )
    {
        Set((IwSize)point, (IwSize)point);
    }

    // Constructor. Sets position to specified values
    IwiPoint(
        IwSize _x, // X coordinate of point
        IwSize _y  // Y coordinate of point
    )
    {
        Set(_x, _y);
    }

    // Constructor from IppiPoint structure
    IwiPoint(
        IppiPoint point // IppiPoint structure
    )
    {
        Set(point.x, point.y);
    }

    // Constructor from IppiPointL structure
    IwiPoint(
        IppiPointL point // IppiPointL structure
    )
    {
        Set(point.x, point.y);
    }

    // Constructor from IppiRect structure
    IwiPoint(
        IppiRect  rect // IppiRect structure
    )
    {
        Set(rect.x, rect.y);
    }

    // Constructor from C IwiRoi structure
    IwiPoint(
        ::IwiRoi rect // C IwiRoi structure
    )
    {
        Set(rect.x, rect.y);
    }

    // Sets position to specified values
    void Set(
        IwSize _x, // X coordinate of point
        IwSize _y  // Y coordinate of point
    )
    {
        x = _x;
        y = _y;
    }

    // IwiPoint to IppiPoint cast operator
    inline operator IppiPoint()  const { IppiPoint point = {(int)x, (int)y}; return point; }
};

// Stores the geometric position and size of a rectangle. Extends IppiRect structure
class IwiRoi: public ::IwiRoi
{
public:
    // Default constructor. Sets values to zero
    IwiRoi()
    {
        Set(0, 0, 0, 0);
    }

    // One value template constructor. Sets position to zero and size to same value. Useful for simple initialization, e.g.: rect = 0
    template<typename T>
    IwiRoi(
        T size // Size of rectangle
    )
    {
        Set(0, 0, (IwSize)size, (IwSize)size);
    }

    // Constructor. Sets rectangle to specified values
    IwiRoi(
        IwSize _x,     // X coordinate of rectangle
        IwSize _y,     // Y coordinate of rectangle
        IwSize _width, // Width of rectangle
        IwSize _height // Height of rectangle
    )
    {
        Set(_x, _y, _width, _height);
    }

    // Constructor from IppiSize structure. Sets position to 0 and size to IppiSize value
    IwiRoi(
        IppiSize size
    )
    {
        Set(0, 0, size.width, size.height);
    }

    // Constructor from C IwiSize structure. Sets position to 0 and size to IwiSize value
    IwiRoi(
        ::IwiSize size
    )
    {
        Set(0, 0, size.width, size.height);
    }

    // Constructor from IwiSize class. Sets position to 0 and size to IwiSize value
    IwiRoi(
        IwiSize size
    )
    {
        Set(0, 0, size.width, size.height);
    }

    // Constructor from IppiRect class
    IwiRoi(
        IppiRect rect
    )
    {
        Set(rect.x, rect.y, rect.width, rect.height);
    }

    // Constructor from C IwiRoi structure
    IwiRoi(
        ::IwiRoi rect
    )
    {
        Set(rect.x, rect.y, rect.width, rect.height);
    }

    // Sets rectangle to specified values
    void Set(
        IwSize _x,     // X coordinate of rectangle
        IwSize _y,     // Y coordinate of rectangle
        IwSize _width, // Width of rectangle
        IwSize _height // Height of rectangle
    )
    {
        x = _x;
        y = _y;
        width  = _width;
        height = _height;
    }

    // Retrns size of area covered by IwiRoi
    inline IwSize Area() const
    {
        return IPP_ABS(this->width*this->height);
    }

    // IwiRoi to IwiPoint cast operator
    inline operator IwiPoint() const { return IwiPoint(x, y); }

    // IwiRoi to IwiSize cast operator
    inline operator IwiSize()  const { return IwiSize(width, height); }

    // IwiRoi to IppiPoint cast operator
    inline operator IppiPoint() const { IppiPoint point = {(int)x, (int)y}; return point; }

    // IwiRoi to IppiSize cast operator
    inline operator IppiSize()  const { IppiSize  size  = {(int)width, (int)height}; return size; }

    // IwiRoi to IppiPointL cast operator
    inline operator IppiPointL() const { IppiPointL point = {x, y}; return point; }

    // IwiRoi to C IwiSize cast operator
    inline operator ::IwiSize()  const { ::IwiSize size  = {width, height}; return size; }
};

// Stores extrapolation type of border and border value for constant border
class IwiBorderType
{
public:
    // Default constructor
    IwiBorderType()
    {
        m_type  = ippBorderRepl;
    }

    // Default constructor with border type
    IwiBorderType(::IwiBorderType borderType)
    {
        m_type  = borderType;
    }

    // Constructor for borders combination
    IwiBorderType(int borderType)
    {
        m_type  = (::IwiBorderType)(borderType);
    }

    // Default constructor with border type and value
    IwiBorderType(::IwiBorderType borderType, IwValueFloat value)
    {
        m_value = value;
        m_type  = borderType;
    }

    // Set new border type without affecting flags
    inline void SetType(::IwiBorderType type)
    {
        m_type = (::IwiBorderType)(StripType()|(0xF&type));
    }

    // Set new border flags without affecting the type
    inline void SetFlags(int flags)
    {
        m_type = (::IwiBorderType)(StripFlags()|(flags&(~0xF)));
    }

    // Return border flags without type
    inline int StripType() const
    {
        return (m_type&(~0xF));
    }

    // Return border type without flags
    inline IwiBorderType StripFlags() const
    {
        return (m_type&0xF);
    }

    // IwiBorderType to bool cast operator
    inline operator bool() const { return (m_type)?true:false; }

    // IwiBorderType to int cast operator
    inline operator int() const { return (int)m_type; }

    // IwiBorderType to ::IwiBorderType cast operator
    inline operator ::IwiBorderType() const { return m_type; }

    // Compares border type
    bool operator==(const IwiBorderType& rhs) const
    {
        if(this->m_type == rhs.m_type)
        {
            if(this->m_type == ippBorderConst)
            {
                if(this->m_value == rhs.m_value)
                    return true;
                else
                    return false;
            }
            return true;
        }
        else
            return false;
    }
    bool operator!=(const IwiBorderType& rhs) const
    {
        return !(*this==rhs);
    }
    bool operator==(const ::IwiBorderType& rhs) const
    {
        if(this->m_type == rhs)
            return true;
        else
            return false;
    }
    bool operator!=(const ::IwiBorderType& rhs) const
    {
        return !(*this==rhs);
    }
    bool operator!() const
    {
        return !(m_type);
    }

    // Computes logical OR for border type. This affects only flags part of the variable
    inline IwiBorderType& operator|=(const int &rhs)
    {
        this->m_type = (IwiBorderType)(this->m_type|(rhs&(~0xF)));
        return *this;
    }
    inline IwiBorderType operator|(const int &rhs) const
    {
        IwiBorderType result = *this;
        result |= rhs;
        return result;
    }
    inline IwiBorderType& operator|=(const IwiBorderType &rhs)
    {
        this->m_type = (IwiBorderType)(this->m_type|(rhs.m_type&(~0xF)));
        return *this;
    }
    inline IwiBorderType operator|(const IwiBorderType &rhs) const
    {
        IwiBorderType result = *this;
        result |= rhs;
        return result;
    }
    inline IwiBorderType& operator|=(const IppiBorderType &rhs)
    {
        this->m_type = (IwiBorderType)(this->m_type|(rhs&(~0xF)));
        return *this;
    }
    inline IwiBorderType operator|(const IppiBorderType &rhs) const
    {
        IwiBorderType result = *this;
        result |= rhs;
        return result;
    }

    // Computes logical AND for border type. This affects only flags part of the variable
    inline IwiBorderType& operator&=(const int &rhs)
    {
        this->m_type = (IwiBorderType)(this->m_type&(rhs&(~0xF)));
        return *this;
    }
    inline IwiBorderType operator&(const int &rhs) const
    {
        IwiBorderType result = *this;
        result &= rhs;
        return result;
    }
    inline IwiBorderType& operator&=(const IwiBorderType &rhs)
    {
        this->m_type = (IwiBorderType)(this->m_type&(rhs.m_type&(~0xF)));
        return *this;
    }
    inline IwiBorderType operator&(const IwiBorderType &rhs) const
    {
        IwiBorderType result = *this;
        result &= rhs;
        return result;
    }
    inline IwiBorderType& operator&=(const IppiBorderType &rhs)
    {
        this->m_type = (IwiBorderType)(this->m_type&(rhs&(~0xF)));
        return *this;
    }
    inline IwiBorderType operator&(const IppiBorderType &rhs) const
    {
        IwiBorderType result = *this;
        result &= rhs;
        return result;
    }

    ::IwiBorderType m_type;
    IwValueFloat    m_value;
};

// Stores border size data
class IwiBorderSize: public ::IwiBorderSize
{
public:
    // Default constructor. Sets values to zero
    IwiBorderSize()
    {
        Set(0, 0, 0, 0);
    }

    // One value template constructor. Sets border to same value. Useful for simple initialization, e.g.: border = 0
    template<typename T>
    IwiBorderSize(
        T border // Position of point
    )
    {
        Set((IwSize)border, (IwSize)border, (IwSize)border, (IwSize)border);
    }

    // Constructor. Sets border to the specified values
    IwiBorderSize(
        IwSize _left,   // Size of border to the left
        IwSize _top,    // Size of border to the top
        IwSize _right,  // Size of border to the right
        IwSize _bottom  // Size of border to the bottom
    )
    {
        Set(_left, _top, _right, _bottom);
    }

    // Constructor from C IwiBorderSize structure
    IwiBorderSize(
        ::IwiBorderSize border // IwiBorderSize structure
    )
    {
        Set(border.left, border.top, border.right, border.bottom);
    }

    // Constructor from IppiBorderSize structure
    IwiBorderSize(
        IppiBorderSize border // IwiBorderSize structure
    )
    {
        Set(border.borderLeft, border.borderTop, border.borderRight, border.borderBottom);
    }

    // Constructor from the image ROI
    IwiBorderSize(
        IwiSize imageSize,   // Size of the image
        IwiRoi  imageRoi     // Image ROI
    )
    {
        Set(imageSize, imageRoi);
    }

    // Constructor from the mask size enumerator
    IwiBorderSize(
        IppiMaskSize mask   // Processing mask size enumerator
    )
    {
        IwiSize size = ::iwiMaskToSize(mask);
        Set(size.width/2, size.height/2, size.width/2, size.height/2);
    }

    // Constructor from the kernel size
    IwiBorderSize(
        IwiSize kernel   // Processing kernel size
    )
    {
        Set(kernel.width/2, kernel.height/2, kernel.width/2, kernel.height/2);
    }

    // Sets border to the specified values
    void Set(
        IwSize _left,   // Size of border to the left
        IwSize _top,    // Size of border to the top
        IwSize _right,  // Size of border to the right
        IwSize _bottom  // Size of border to the bottom
    )
    {
        left      = _left;
        top       = _top;
        right     = _right;
        bottom    = _bottom;
    }

    // Sets border from the image ROI
    void Set(
        IwiSize imageSize,   // Size of the image
        IwiRoi  imageRoi     // Image ROI
    )
    {
        left      = imageRoi.x;
        top       = imageRoi.y;
        right     = imageSize.width  - imageRoi.x - imageRoi.width;
        bottom    = imageSize.height - imageRoi.y - imageRoi.height;
    }

    // Returns true if all borders are zero
    bool Empty() const
    {
        return !(this->left || this->top || this->right || this->bottom);
    }

    // Returns border size which contains maximum values of two border
    static IwiBorderSize Max(IwiBorderSize lhs, const IwiBorderSize &rhs)
    {
        lhs.left    = IPP_MAX(lhs.left, rhs.left);
        lhs.top     = IPP_MAX(lhs.top, rhs.top);
        lhs.right   = IPP_MAX(lhs.right, rhs.right);
        lhs.bottom  = IPP_MAX(lhs.bottom, rhs.bottom);
        return lhs;
    }

    // Returns border size which contains minimum values of two border
    static IwiBorderSize Min(IwiBorderSize lhs, const IwiBorderSize &rhs)
    {
        lhs.left    = IPP_MIN(lhs.left, rhs.left);
        lhs.top     = IPP_MIN(lhs.top, rhs.top);
        lhs.right   = IPP_MIN(lhs.right, rhs.right);
        lhs.bottom  = IPP_MIN(lhs.bottom, rhs.bottom);
        return lhs;
    }

    // Adds constant to the border
    inline IwiBorderSize& operator+=(const int &rhs)
    {
        this->left   += rhs;
        this->top    += rhs;
        this->right  += rhs;
        this->bottom += rhs;
        return *this;
    }
    inline IwiBorderSize operator+(const int &rhs) const
    {
        IwiBorderSize result = *this;
        result += rhs;
        return result;
    }

    // Subtracts constant from the border
    inline IwiBorderSize& operator-=(const int &rhs)
    {
        this->left   -= rhs;
        this->top    -= rhs;
        this->right  -= rhs;
        this->bottom -= rhs;
        return *this;
    }
    inline IwiBorderSize operator-(const int &rhs) const
    {
        IwiBorderSize result = *this;
        result -= rhs;
        return result;
    }

    // Multiplies the border by the constant
    inline IwiBorderSize& operator*=(const double &rhs)
    {
        this->left   = (IwSize)(this->left*rhs);
        this->top    = (IwSize)(this->top*rhs);
        this->right  = (IwSize)(this->right*rhs);
        this->bottom = (IwSize)(this->bottom*rhs);
        return *this;
    }
    inline IwiBorderSize operator*(const double &rhs) const
    {
        IwiBorderSize result = *this;
        result *= rhs;
        return result;
    }

    // Divides the border by the constant
    inline IwiBorderSize& operator/=(const double &rhs)
    {
        this->left   = (IwSize)(this->left/rhs);
        this->top    = (IwSize)(this->top/rhs);
        this->right  = (IwSize)(this->right/rhs);
        this->bottom = (IwSize)(this->bottom/rhs);
        return *this;
    }
    inline IwiBorderSize operator/(const double &rhs) const
    {
        IwiBorderSize result = *this;
        result /= rhs;
        return result;
    }

    // Adds border to the border
    inline IwiBorderSize& operator+=(const IwiBorderSize &rhs)
    {
        this->left   += rhs.left;
        this->top    += rhs.top;
        this->right  += rhs.right;
        this->bottom += rhs.bottom;
        return *this;
    }
    inline IwiBorderSize operator+(const IwiBorderSize &rhs) const
    {
        IwiBorderSize result = *this;
        result += rhs;
        return result;
    }

    // Subtracts border from the border
    inline IwiBorderSize& operator-=(const IwiBorderSize &rhs)
    {
        this->left   -= rhs.left;
        this->top    -= rhs.top;
        this->right  -= rhs.right;
        this->bottom -= rhs.bottom;
        return *this;
    }
    inline IwiBorderSize operator-(const IwiBorderSize &rhs) const
    {
        IwiBorderSize result = *this;
        result -= rhs;
        return result;
    }

};

// Convert IppiMaskSize enumerator to actual IwiSize size
// Returns:
//      Width and height of IppiMaskSize in pixels
IW_DECL_CPP(IwiSize) iwiMaskToSize(
    IppiMaskSize mask    // Kernel or mask size enumerator
)
{
    return ::iwiMaskToSize(mask);
}

// Convert kernel or mask size to border size
// Returns:
//      Border required for a filter with specified kernel size
IW_DECL_CPP(IwiBorderSize) iwiSizeToBorderSize(
    IwiSize kernelSize   // Size of kernel as from iwiMaskToSize() or arbitrary
)
{
    return ::iwiSizeToBorderSize(kernelSize);
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiImage - Image class
///////////////////////////////////////////////////////////////////////////// */

// IwiImage is a base class for IW image processing functions to store input and output data.
class IwiImage: public ::IwiImage
{
public:
    // Default constructor. Sets values to zero
    IwiImage()
    {
        iwiImage_Init(this);

        m_pRefCounter = new int;
        if(!m_pRefCounter)
            OWN_ERROR_THROW_ONLY(ippStsMemAllocErr)
        (*m_pRefCounter) = 1;
    }

    // Copy constructor for C++ object. Performs lazy copy of an internal image
    IwiImage(
        const IwiImage &image           // Source image
    )
    {
        iwiImage_Init(this);

        m_pRefCounter = new int;
        if(!m_pRefCounter)
            OWN_ERROR_THROW_ONLY(ippStsMemAllocErr)
        (*m_pRefCounter) = 1;

        *this = image;
    }

    // Copy constructor for C object. Initializes image structure with external buffer
    IwiImage(
        const ::IwiImage &image         // Source image
    )
    {
        iwiImage_Init(this);

        m_pRefCounter = new int;
        if(!m_pRefCounter)
            OWN_ERROR_THROW_ONLY(ippStsMemAllocErr)
        (*m_pRefCounter) = 1;

        *this = image;
    }

    // Constructor with initialization. Initializes image structure with external buffer
    IwiImage(
        IwiSize         size,                           // Image size, in pixels
        IppDataType     dataType,                       // Image pixel type
        int             channels,                       // Number of image channels
        IwiBorderSize   inMemBorder = IwiBorderSize(),  // Size of border around image or NULL if there is no border
        void           *pBuffer     = NULL,             // Pointer to the external buffer image buffer
        IwSize          step        = 0                 // Distance, in bytes, between the starting points of consecutive lines in the external buffer
    )
    {
        IppStatus status;
        iwiImage_Init(this);

        m_pRefCounter = new int;
        if(!m_pRefCounter)
            OWN_ERROR_THROW_ONLY(ippStsMemAllocErr)
        (*m_pRefCounter) = 1;

        status = Init(size, dataType, channels, inMemBorder, pBuffer, step);
        OWN_ERROR_CHECK_THROW_ONLY(status);
    }

    // Default destructor
    ~IwiImage()
    {
        Release();

        if(m_pRefCounter)
        {
            if(iwAtomic_AddInt(m_pRefCounter, -1) == 1)
                delete m_pRefCounter;
        }
    }

    // Copy operator for C++ object. Performs lazy copy of an internal image
    IwiImage& operator=(const IwiImage &image)
    {
        if(!m_pRefCounter)
            OWN_ERROR_THROW_ONLY(ippStsNullPtrErr)

        if(&image == this)
            return *this;

        if(image.m_ptr)
        {
            IppStatus status = Init(image.m_size, image.m_dataType, image.m_channels, image.m_inMemSize, image.m_ptr, image.m_step);
            OWN_ERROR_CHECK_THROW_ONLY(status);
        }
        else
        {
            IppStatus status = Init(image.m_size, image.m_dataType, image.m_channels, image.m_inMemSize, image.m_ptrConst, image.m_step);
            OWN_ERROR_CHECK_THROW_ONLY(status);
        }

        if(image.m_pBuffer)
        {
            iwAtomic_AddInt(image.m_pRefCounter, 1);

            if(iwAtomic_AddInt(m_pRefCounter, -1) == 1)
                delete this->m_pRefCounter;
            this->m_pRefCounter = image.m_pRefCounter;
            this->m_pBuffer     = image.m_pBuffer;
        }

        return *this;
    }

    // Copy operator for C object. Initializes image structure with external buffer
    IwiImage& operator=(const ::IwiImage &image)
    {
        if(!m_pRefCounter)
            OWN_ERROR_THROW_ONLY(ippStsNullPtrErr)

        if(image.m_ptr)
        {
            IppStatus status = Init(image.m_size, image.m_dataType, image.m_channels, image.m_inMemSize, image.m_ptr, image.m_step);
            OWN_ERROR_CHECK_THROW_ONLY(status);
        }
        else
        {
            IppStatus status = Init(image.m_size, image.m_dataType, image.m_channels, image.m_inMemSize, image.m_ptrConst, image.m_step);
            OWN_ERROR_CHECK_THROW_ONLY(status);
        }
        return *this;
    }

    // Initializes image structure with external buffer
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus Init(
        IwiSize         size,                           // Image size, in pixels
        IppDataType     dataType,                       // Image pixel type
        int             channels,                       // Number of image channels
        IwiBorderSize   inMemBorder = IwiBorderSize(),  // Size of border around image or NULL if there is no border
        void           *pBuffer     = NULL,             // Pointer to the external buffer image buffer
        IwSize          step        = 0                 // Distance, in bytes, between the starting points of consecutive lines in the external buffer
    )
    {
        if(this->m_pBuffer && this->m_pBuffer == pBuffer)
            return ippStsNoErr;

        IppStatus status = Release();
        OWN_ERROR_CHECK(status);

        status = iwiImage_InitExternal(this, size, dataType, channels, &inMemBorder, pBuffer, step);
        OWN_ERROR_CHECK(status);
        return status;
    }

    // Initializes image structure with external read-only buffer
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus Init(
        IwiSize         size,                           // Image size, in pixels
        IppDataType     dataType,                       // Image pixel type
        int             channels,                       // Number of image channels
        IwiBorderSize   inMemBorder,                    // Size of border around image or NULL if there is no border
        const void     *pBuffer,                        // Pointer to the external buffer image buffer
        IwSize          step                            // Distance, in bytes, between the starting points of consecutive lines in the external buffer
    )
    {
        if(this->m_pBuffer && this->m_pBuffer == pBuffer)
            return ippStsNoErr;

        IppStatus status = Release();
        OWN_ERROR_CHECK(status);

        status = iwiImage_InitExternalConst(this, size, dataType, channels, &inMemBorder, pBuffer, step);
        OWN_ERROR_CHECK(status);
        return status;
    }

    // Initializes image structure and allocates image data
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus Alloc(
        IwiSize         size,                           // Image size, in pixels
        IppDataType     dataType,                       // Image pixel type
        int             channels,                       // Number of image channels
        IwiBorderSize   inMemBorder = IwiBorderSize()   // Size of border around image or NULL if there is no border
    )
    {
        IppStatus status = Release();
        OWN_ERROR_CHECK(status);

        status = iwiImage_Alloc(this, size, dataType, channels, &inMemBorder);
        OWN_ERROR_CHECK(status);
        return status;
    }

    // Releases image data if it was allocated by IwiImage::Alloc
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus Release()
    {
        if(!m_pRefCounter)
            OWN_ERROR_THROW_ONLY(ippStsNullPtrErr)

        if(iwAtomic_AddInt(m_pRefCounter, -1) > 1)
        {
            m_pRefCounter    = new int;
            (*m_pRefCounter) = 1;

            m_pBuffer = NULL;
            m_ptr     = NULL;
            m_step    = 0;
        }
        else
        {
            (*m_pRefCounter) = 1;
            iwiImage_Release(this);
        }
        return ippStsNoErr;
    }

    // Returns pointer to specified pixel position in image buffer
    // Returns:
    //      Pointer to the image data
    inline void* ptr(
        IwSize y  = 0,  // y shift, as rows
        IwSize x  = 0,  // x shift, as columns
        int    ch = 0   // channels shift
    ) const
    {
        return iwiImage_GetPtr(this, y, x, ch);
    }

    // Returns pointer to specified pixel position in read-only image buffer
    // Returns:
    //      Pointer to the image data
    inline const void* ptrConst(
        IwSize y  = 0,  // y shift, as rows
        IwSize x  = 0,  // x shift, as columns
        int    ch = 0   // channels shift
    ) const
    {
        return iwiImage_GetPtrConst(this, y, x, ch);
    }

    // Applies ROI to the current image by adjusting size and starting point of the image. Can be applied recursively.
    // This function saturates ROIs which step outside of the image border.
    // If ROI has no intersection with the image then resulted image size will be 0x0
    // Throws:
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus RoiSet(
        ipp::IwiRoi roi // Roi rectangle of the required sub-image
    )
    {
        IppStatus status = iwiImage_RoiSet(this, roi);
        OWN_ERROR_CHECK(status);
        return ippStsNoErr;
    }

    // Returns sub-image with size and starting point of the specified ROI
    // Returns:
    //      IwiImage object of sub-image
    IwiImage GetRoiImage(
        ipp::IwiRoi roi             // Roi rectangle of the required sub-image
    ) const
    {
        return iwiImage_GetRoiImage(this, roi);
    }

    // Add border size to current inMem image border, making image size smaller. Resulted image cannot be smaller than 1x1 pixels
    // Throws:
    //      ippStsSizeErr                       ROI size is illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    inline IwiImage& operator+=(const IwiBorderSize &rhs)
    {
        IppStatus status = iwiImage_BorderAdd(this, rhs);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }
    inline IwiImage operator+(const IwiBorderSize &rhs) const
    {
        IwiImage  result = *this;
        IppStatus status = iwiImage_BorderAdd(&result, rhs);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return result;
    }

    // Subtracts border size from current inMem image border, making image size bigger. Resulted border cannot be lesser than 0
    // Throws:
    //      ippStsOutOfRangeErr                 ROI is out of image
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator-=(const IwiBorderSize &rhs)
    {
        IppStatus status = iwiImage_BorderSub(this, rhs);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }
    inline IwiImage operator-(const IwiBorderSize &rhs) const
    {
        IwiImage  result = *this;
        IppStatus status = iwiImage_BorderSub(&result, rhs);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return result;
    }

    // Set border size to current inMem image border, adjusting image size. Resulted image cannot be smaller than 1x1 pixels.
    // Throws:
    //      ippStsSizeErr                       ROI size is illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator=(const IwiBorderSize &rhs)
    {
        IppStatus status = iwiImage_BorderSet(this, rhs);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

    // Returns true if image has an assigned buffer
    inline bool Exists() const
    {
        return (this->m_ptrConst)?true:false;
    }

    // Returns true if image doesn't have an assigned buffer or its dimensions have zero size
    inline bool Empty() const
    {
        return (Exists() && this->m_size.width && this->m_size.height)?false:true;
    }

    // Compares image structures and returns true if structure parameters are compatible, e.g. copy operation can be performed without reallocation
    bool Similar(const ipp::IwiImage& rhs) const
    {
        if(this->m_dataType == rhs.m_dataType &&
            this->m_channels == rhs.m_channels)
            return true;
        else
            return false;
    }

    /**/////////////////////////////////////////////////////////////////////////////
    //                   Arithmetic operators
    ///////////////////////////////////////////////////////////////////////////// */

    // Adds one image to another
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator+=(const IwiImage &rhs)
    {
        IppStatus status = ::iwiAdd(&rhs, this, this, NULL, NULL);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

    // Adds constant to the image
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator+=(const IwValueFloat &rhs)
    {
        IppStatus status = ::iwiAddC(rhs, rhs.ValuesNum(), this, this, NULL, NULL);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

    // Subtracts one image from another
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator-=(const IwiImage &rhs)
    {
        IppStatus status = ::iwiSub(&rhs, this, this, NULL, NULL);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

    // Subtracts constant from the image
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator-=(const IwValueFloat &rhs)
    {
        IppStatus status = ::iwiSubC(rhs, rhs.ValuesNum(), this, this, NULL, NULL);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

    // Multiplies one image by another
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator*=(const IwiImage &rhs)
    {
        IppStatus status = ::iwiMul(&rhs, this, this, NULL, NULL);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

    // Multiplies the image by the constant
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator*=(const IwValueFloat &rhs)
    {
        IppStatus status = ::iwiMulC(rhs, rhs.ValuesNum(), this, this, NULL, NULL);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

    // Divides one image by another
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator/=(const IwiImage &rhs)
    {
        IppStatus status = ::iwiDiv(&rhs, this, this, NULL, NULL);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

    // Divides the image by the constant
    // Throws:
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    inline IwiImage& operator/=(const IwValueFloat &rhs)
    {
        IppStatus status = ::iwiDivC(rhs, rhs.ValuesNum(), this, this, NULL, NULL);
        OWN_ERROR_CHECK_THROW_ONLY(status);
        return *this;
    }

private:
    int *m_pRefCounter;  // Shared reference counter for allocated memory
};

// IwiImageArray holds an array of IwiImages for processing
class IwiImageArray
{
public:
    // Default constructor
    IwiImageArray(
        const IwiImage &image1 = IwiImage(),
        const IwiImage &image2 = IwiImage(),
        const IwiImage &image3 = IwiImage(),
        const IwiImage &image4 = IwiImage()
    )
    {
        imArray[0] = image1;
        imArray[1] = image2;
        imArray[2] = image3;
        imArray[3] = image4;
    }

    // Copy operator for single IwiImage object
    const IwiImageArray& operator=(const IwiImage &image)
    {
        imArray[0] = image;
        return *this;
    }

    IwiImage imArray[4];
};


/* /////////////////////////////////////////////////////////////////////////////
//                   IW Tiling
///////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiTile-based basic tiling
///////////////////////////////////////////////////////////////////////////// */

// This is a wrapper class for the basic IwiTile tiling API
class IwiTile: public ::IwiTile
{
public:
    // Default constructor.
    IwiTile()
    {
        this->m_initialized = 0;
    }

    // Constructor with initialization.
    IwiTile(
        const ::IwiRoi &tileRoi    // [in] Tile offset and size
    )
    {
        this->m_initialized = 0;
        SetRoi(tileRoi);
    }

    // Basic tiling initializer for IwiTile structure.
    // Use this method to set up single function tiling or tiling for pipelines with border-less functions.
    // For functions which operate with different sizes for source and destination images use destination size as a base
    // for tile parameters.
    void SetRoi(
        const ::IwiRoi &tileRoi    // [in] Tile offset and size
    )
    {
        *(::IwiTile*)this = ::iwiTile_SetRoi(tileRoi);
    }

    // Assignment operator from C IwiRoi structure.
    IwiTile& operator=(
        const ::IwiRoi &tileRoi    // [in] Tile offset and size
    )
    {
        SetRoi(tileRoi);
        return *this;
    }

/* /////////////////////////////////////////////////////////////////////////////
//                   Manual tiling control
///////////////////////////////////////////////////////////////////////////// */

    // Returns border with proper ippBorderInMem flags for current tile position, image size and border size
    // Returns:
    //      ippBorderInMem flags
    static IwiBorderType GetTileBorder(
        IwiRoi          roi,            // Tile position and size
        ::IwiBorderType border,         // Border type
        IwiBorderSize   borderSize,     // Border size
        IwiSize         srcImageSize    // Source image size
    )
    {
        return ::iwiTile_GetTileBorder(roi, border, borderSize, srcImageSize);
    }

    // Returns minimal acceptable tile size for the current border size and type
    // Returns:
    //      Minimal tile size
    static IwiSize GetMinTileSize(
        ::IwiBorderType border,     // Border type
        IwiBorderSize   borderSize  // Border size
    )
    {
        return ::iwiTile_GetMinTileSize(border, borderSize);
    }

    // Function corrects ROI position and size to prevent overlapping between filtering function border and image border in
    // case of border reconstruction. If image already has a right or a bottom border in memory and border type flags
    // ippBorderInMemRight or ippBorderInMemBottom were specified accordingly then no correction is required.
    //
    // C API descriptions has more details.
    // Returns:
    //      Corrected ROI
    static IwiRoi CorrectBordersOverlap(
        IwiRoi          tile,           // [in]     Tile position and size to be checked and corrected
        ::IwiBorderType border,         // [in]     Border type
        IwiBorderSize   borderSize,     // [in]     Border size
        IwiSize         srcImageSize    // [in]     Source image size
    )
    {
        return ::iwiTile_CorrectBordersOverlap(tile, border, borderSize, srcImageSize);
    }
};

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiTile-based pipeline tiling
///////////////////////////////////////////////////////////////////////////// */

// Important notice:
// This tiling API is created for tiling of complex pipelines with functions which use borders.
// Tiling of pipelines instead of isolated functions can increase scalability of threading or performance of
// non-threaded functions by performing all operations inside CPU cache.
//
// This is advanced tiling method, so you better know what you are doing.
// 1. Pipeline tiling operates in reverse order: from destination to source.
//    a. Use tile size based on final destination image size
//    b. Initialize IwiTile structure with iwiTilePipeline_Init for the last operation
//    c. Initialize IwiTile structure for other operations from last to first with iwiTilePipeline_InitChild
// 2. Derive border size for each operation from its mask size, kernel size or specific border size getter if any
// 3. If you have geometric transform inside pipeline, you need to implement IwiTileTransform interface for IwiTile for this transform operation
// 4. In case of threading don't forget to copy initialized IwiTile structures to local thread or initialize them on
//    per-thread basis. Access to structures is not thread safe!
// 5. Do not exceed maximum tile size specified during initialization. This can lead to buffers overflow!
//
// There is a set of examples covering usage of tiling. Please refer to them for help.

// Transform proxy function to loop-the-loop around interfaces an allow to use proper C++ approach. Don't use it directly.
static int IPP_STDCALL __proxy_getSrcRoi__(::IwiRoi dstRoi, ::IwiRoi *pSrcRoi, void* pParams);

// Geometric transform interface
// To use geometric transforms inside pipeline tiling you need to implement this interface.
class IwiTileTransform
{
public:
    IwiTileTransform()
    {
        transformStruct.getSrcRoiFun = __proxy_getSrcRoi__;
        transformStruct.pParams      = this;
        transformStruct.srcImageSize = IwiSize();
    }
    virtual ~IwiTileTransform() {}

    // Source ROI getter
    virtual int GetSrcRoi(ipp::IwiRoi, ipp::IwiRoi&) const = 0;

    // Object clone method. This method is required to properly save transform object for current tile
    virtual void* Clone() const = 0;

    // IwiTileTransform to ::IwiTileTransform* cast operator
    virtual operator const ::IwiTileTransform*() const { return &transformStruct; }

protected:
    // Clone helper. This method can be use to implement Clone method for particular type
    template<class T>
    T* CloneT() const
    {
        T *clone = new T(*((T*)this));
        clone->transformStruct.pParams = clone; // Switch parameters for proxy function to address of new object
        return clone;
    }

    ::IwiTileTransform transformStruct;
};

// IwiTileTransform implementation for operation without transform.
class IwiTileNoTransform: public IwiTileTransform
{
public:
    IwiTileNoTransform()
    {
        transformStruct.getSrcRoiFun = NULL;
        transformStruct.pParams      = NULL;
    }
    virtual ~IwiTileNoTransform() {}

    // IwiTileTransform to ::IwiTileTransform* cast operator
    virtual operator const ::IwiTileTransform*() const { return NULL; }

    virtual void* Clone() const
    {
        return NULL;
    }

private:
    // Source ROI getter
    virtual int GetSrcRoi(ipp::IwiRoi, ipp::IwiRoi&) const
    {
        return -1;
    }
};

// Transform proxy function
static int IPP_STDCALL __proxy_getSrcRoi__(::IwiRoi dstRoi, ::IwiRoi *pSrcRoi, void* pParams)
{
    if(!pSrcRoi || !pParams)
        return -1;

    ipp::IwiTileTransform *pTrans = (ipp::IwiTileTransform*)pParams;
    ipp::IwiRoi srcRoi;
    int status = pTrans->GetSrcRoi(dstRoi, srcRoi);
    if(!status)
        *pSrcRoi = srcRoi;
    return status;
}

// This is a wrapper class for the pipeline IwiTile tiling API
class IwiTilePipeline: public IwiTile
{
public:
    // Default constructor.
    IwiTilePipeline()
    {
        this->pTransform    = NULL;
        this->m_initialized = 0;
    }

    // Constructor with initialization for the root node.
    // Throws:
    //      ippStsBadArgErr                     incorrect arg/param of the function
    //      ippStsNullPtrErr                    unexpected NULL pointer
    IwiTilePipeline(
        IwiSize                 tileSizeMax,                                // [in] Maximum tile size for intermediate buffers size calculation
        IwiSize                 dstImageSize,                               // [in] Destination image size for current operation
        const IwiBorderType    &borderType        = IwiBorderType(),        // [in] Border type for the current operation
        const IwiBorderSize    &borderSize        = IwiBorderSize(),        // [in] Border size for the current operation
        const IwiTileTransform &transformStruct   = IwiTileNoTransform()    // [in] Initialized transform structure if operation performs geometric transformation
    )
    {
        this->pTransform    = NULL;
        this->m_initialized = 0;
        IppStatus status = Init(tileSizeMax, dstImageSize, borderType, borderSize, transformStruct);
        OWN_ERROR_CHECK_THROW_ONLY(status);
    }

    // Constructor with initialization for the child node.
    // Throws:
    //      ippStsBadArgErr                     incorrect arg/param of the function
    //      ippStsNullPtrErr                    unexpected NULL pointer
    IwiTilePipeline(
        IwiTilePipeline        &parent,                                     // [in] IwiTile structure of previous operation
        const IwiBorderType    &borderType        = IwiBorderType(),        // [in] Border type for the current operation
        const IwiBorderSize    &borderSize        = IwiBorderSize(),        // [in] Border size for the current operation
        const IwiTileTransform &transformStruct   = IwiTileNoTransform()    // [in] Initialized transform structure if operation performs geometric transformation
    ) : IwiTile()
    {
        this->pTransform    = NULL;
        this->m_initialized = 0;
        IppStatus status = InitChild(parent, borderType, borderSize, transformStruct);
        OWN_ERROR_CHECK_THROW_ONLY(status);
    }

    // Default destructor
    ~IwiTilePipeline()
    {
        iwiTilePipeline_Release(this);
        if(this->pTransform)
            delete this->pTransform;
    }

    // Pipeline tiling root node initializer for IwiTile structure.
    // This initializer should be used first and for IwiTile structure of the final operation.
    // Throws:
    //      ippStsBadArgErr                     incorrect arg/param of the function
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus Init(
        IwiSize                 tileSizeMax,                        // [in] Maximum tile size for intermediate buffers size calculation
        IwiSize                 dstImageSize,                       // [in] Destination image size for current operation
        const IwiBorderType    &borderType  = IwiBorderType(),      // [in] Border type for the current operation
        const IwiBorderSize    &borderSize  = IwiBorderSize(),      // [in] Border size for the current operation
        const IwiTileTransform &transform   = IwiTileNoTransform()  // [in] Initialized transform structure if operation performs geometric transformation
    )
    {
        ::IwiBorderType border = borderType;
        if(this->pTransform)
            delete this->pTransform;
        pTransform = (IwiTileTransform*)transform.Clone();
        IppStatus status = ::iwiTilePipeline_Init(this, tileSizeMax, dstImageSize, &border, ((borderSize.Empty())?NULL:&borderSize), ((pTransform)?((const ::IwiTileTransform*)(*pTransform)):(NULL)));
        OWN_ERROR_CHECK(status);
        return status;
    }

    // Pipeline tiling child node initializer for IwiTile structure.
    // This initializer should be called for any operation preceding the last operation in reverse order.
    // Throws:
    //      ippStsBadArgErr                     incorrect arg/param of the function
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus InitChild(
        IwiTilePipeline        &parent,                             // [in] IwiTile structure of previous operation
        const IwiBorderType    &borderType  = IwiBorderType(),      // [in] Border type for the current operation
        const IwiBorderSize    &borderSize  = IwiBorderSize(),      // [in] Border size for the current operation
        const IwiTileTransform &transform   = IwiTileNoTransform()  // [in] Initialized transform structure if operation performs geometric transformation
    )
    {
        ::IwiBorderType border = borderType;
        if(this->pTransform)
            delete this->pTransform;
        pTransform = (IwiTileTransform*)transform.Clone();
        IppStatus status = ::iwiTilePipeline_InitChild(this, &parent, &border, ((borderSize.Empty())?NULL:&borderSize), (pTransform)?((const ::IwiTileTransform*)(*pTransform)):NULL);
        OWN_ERROR_CHECK(status);
        return status;
    }

    // Sets current tile rectangle for the pipeline to process
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus SetTile(
        IwiRoi          tileRoi                // [in] Tile offset and size
    )
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW(ippStsContextMatchErr);

        IppStatus status = ::iwiTilePipeline_SetRoi(this, tileRoi);
        OWN_ERROR_CHECK(status);
        return status;
    }

    // Pipeline tiling intermediate buffer size getter
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    // Returns:
    //      Destination buffer size required by the current pipeline operation
    IwiSize GetDstBufferSize()
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr);

        IwiSize   size;
        IppStatus status = ::iwiTilePipeline_GetDstBufferSize(this, &size);
        OWN_ERROR_CHECK_THROW_ONLY(status);

        return size;
    }

    // Calculates actual border parameter with InMem flags for the current tile absolute and relative offsets and sizes
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    // Returns:
    //      Border with InMem flags actual for the current tile
    IwiBorderType GetTileBorder(
        ::IwiBorderType border                 // [in] Extrapolation algorithm for out of image pixels
    )
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr);

        IppStatus status = ::iwiTilePipeline_GetTileBorder(this, &border);
        OWN_ERROR_CHECK_THROW_ONLY(status);

        return border;
    }

    // This function builds border for the current tile source buffer.
    // This allows to feed function with InMem borders only thus reducing possiblity of borders conflicts on image boundary.
    // By default this function is not applied to the first image in the pipeline, only to intermediate buffers, but
    // it can be used manually to construct border for it too.
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus BuildBorder(
        IwiImage        &srcImage,      // [in,out] Pointer to the source image for which to build border
        IwiBorderType   &border         // [in,out] Extrapolation algorithm for out of image pixels. Updated InMem flags will be returned here
    )
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW(ippStsContextMatchErr);

        IppStatus status = ::iwiTilePipeline_BuildBorder(this, &srcImage, &border.m_type, border.m_value);
        OWN_ERROR_CHECK(status);
        return status;
    };

    // Returns full size of source image for the child pipeline element which includes required InMem borders.
    // This function is required to supply correct image size for geometric transform functions.
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      Full image size with InMem part
    IwiSize GetChildSrcImageSize(
        IwiSize srcOrigSize // [in]     Original source image size
    )
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr);

        IwiSize   size;
        IppStatus status =  ::iwiTilePipeline_GetChildSrcImageSize(this, srcOrigSize, &size);
        OWN_ERROR_CHECK_THROW_ONLY(status);

        return size;
    }

    // Returns full size of destination image for the child pipeline element which includes required InMem borders.
    // This function is required to supply correct image size for geometric transform functions.
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      Full image size with InMem part
    IwiSize GetChildDstImageSize(
        IwiSize dstOrigSize // [in]     Original destination image size
    )
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr);

        IwiSize   size;
        IppStatus status =  ::iwiTilePipeline_GetChildDstImageSize(this, dstOrigSize, &size);
        OWN_ERROR_CHECK_THROW_ONLY(status);

        return size;
    }

    // Checks for image and buffer boundaries for the source buffer and limits tile rectangle
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    // Returns:
    //      Source ROI bounded to the buffer size
    IwiRoi GetBoundedSrcRoi()
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr);

        IwiRoi    roi;
        IppStatus status =  ::iwiTilePipeline_GetBoundedSrcRoi(this, &roi);
        OWN_ERROR_CHECK_THROW_ONLY(status);

        return roi;
    }

    // Checks for image and buffer boundaries for the destination buffer and limits tile rectangle
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    // Returns:
    //      Destination ROI bounded to the buffer size
    IwiRoi GetBoundedDstRoi()
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr);

        IwiRoi    roi;
        IppStatus status =  ::iwiTilePipeline_GetBoundedDstRoi(this, &roi);
        OWN_ERROR_CHECK_THROW_ONLY(status);

        return roi;
    }

    // Returns minimal acceptable tile size for current pipeline
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    //      ippStsErr                           tile calculation error
    // Returns:
    //      Minimal tile size allowed by the pipeline
    IwiSize GetMinTileSize()
    {
        if(!this->m_initialized)
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr);

        IwiSize   minSize;
        IppStatus status = ::iwiTilePipeline_GetMinTileSize(this, &minSize);
        OWN_ERROR_CHECK_THROW_ONLY(status);

        return minSize;
    }

private:
    // Disabled copy operator
    IwiTilePipeline& operator=(const IwiTilePipeline &) { return *this; }

    IwiTileTransform *pTransform;
};

}

#endif
