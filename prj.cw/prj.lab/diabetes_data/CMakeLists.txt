add_library(diabetes_data diabetes_data.cpp include/diabetes_data/diabetes_data.hpp)
set_property(TARGET diabetes_data PROPERTY CXX_STANDARD 20)
target_include_directories(diabetes_data PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS diabetes_data)