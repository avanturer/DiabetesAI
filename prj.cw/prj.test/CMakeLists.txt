add_executable(main main.cpp)
target_link_libraries(main regression)
target_link_libraries(main diabetes_data)
target_link_libraries(main plot)

