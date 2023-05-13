add_library(plot plot.cpp include/plot/plot.hpp)
set_property(TARGET plot PROPERTY CXX_STANDARD 20)
target_include_directories(plot PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS plot)