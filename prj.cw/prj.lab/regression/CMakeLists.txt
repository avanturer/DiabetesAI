add_library(regression regression.cpp include/regression/regression.hpp)
set_property(TARGET regression PROPERTY CXX_STANDARD 20)
target_include_directories(regression PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS regression)