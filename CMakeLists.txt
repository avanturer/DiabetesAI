project(DiabetesAI)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "..." FORCE)
endif()

find_package(Doxygen REQUIRED)

add_subdirectory(prj.cw)

install(TARGETS libtest RUNTIME
        DESTINATION bin
        )

install(DIRECTORY ${CMAKE_SOURCE_DIR}/prj.cw/prj.lab/data/
        DESTINATION data
        )

if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/prj.cw/Doxyfile)
    set(DOXYGEN_CONFIGURED_INPUT ${DOXYGEN_IN})

    add_custom_target(doc ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIGURED_INPUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
            )

endif()

if(DOXYGEN_FOUND)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/prj.cw/doxygen/latex
            DESTINATION docs)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/prj.cw/doxygen/html
            DESTINATION docs)
endif()
