set(PROJECT_NAME BNN)
set(CXX_STANDARD 20)

################################################################################
# Source groups
################################################################################
set(Header_Files
    "BNN.h"
    "Image/Image.h"
    "Image/stb_image.h"
    "Image/stb_image_write.h"
    "Layers/AvgPool.h"
    "Layers/AvgUpool.h"
    "Layers/Conv.h"
    "Layers/Dense.h"
    "Layers/Dropout.h"
    "Layers/Input.h"
    "Layers/Layer.h"
    "Layers/Layers.h"
    "Layers/Output.h"
    "Layers/OutShuf.h"
    "Layers/PixShuf.h"
    "Layers/Resize.h"
    "Layers/SConv.h"
    "Layers/Shape.h"
    "Layers/TConv.h"
    "Misc/Afun.h"
    "Misc/Drafts.h"
    "Misc/Eigen_math.h"
    "Misc/Eigen_util.h"
    "Misc/Misc.h"
    "NNet/GD_nodes.h"
    "NNet/NNet.h"
    "NNet/NNet_samples.h"
    "NNet/Optimizer.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "BNN.cpp"
    "Image/Image.cpp"
    "Misc/Eigen_math.cpp"
    "NNet/NNet.cpp"
    "NNet/NNet_samples.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
set(ROOT_NAMESPACE BNN)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
endif()
################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "_CONSOLE;"
        "UNICODE;"
        "_UNICODE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_CONSOLE;"
        "UNICODE;"
        "_UNICODE"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /Gy
            >
            /permissive-;
            /MP;
            /Gm-;
            /O2;
            /Ob2;
            /Oi;
            /sdl;
            /W3;
            /fp:except-;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            /Ot;
            /fp:fast;
            -Ofast;
            -march=haswell;
            -openmp;
            -fms-extensions;
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /Gy
            >
            /permissive-;
            /MP;
            /O2;
            /Ob2;
            /Oi;
            /sdl;
            /W3;
            /fp:except-;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            /Ot;
            /fp:fast;
            -Ofast;
            -march=haswell;
            -openmp;
            -fms-extensions;
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /SUBSYSTEM:CONSOLE
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /SUBSYSTEM:CONSOLE
        )
    endif()
endif()

################################################################################
# Dependencies
################################################################################
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(BNN PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(${PROJECT_NAME} PUBLIC "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

