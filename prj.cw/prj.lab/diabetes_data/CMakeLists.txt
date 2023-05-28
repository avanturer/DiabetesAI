add_library(diabetes_data diabetes_data.cpp include/diabetes_data/diabetes_data.hpp)
set_property(TARGET diabetes_data PROPERTY CXX_STANDARD 20)
file(COPY ${CMAKE_SOURCE_DIR}/prj.cw/prj.lab/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(diabetes_data PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS diabetes_data)