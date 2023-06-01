add_executable(libtest libtest.cpp)
target_link_libraries(libtest regression)
target_link_libraries(libtest diabetes_data)
target_link_libraries(libtest plot)


