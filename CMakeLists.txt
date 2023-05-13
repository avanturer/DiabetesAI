find_package(Python3 COMPONENTS Development NumPy)
target_include_directories(your_project PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(your_project Python3::Python Python3::NumPy)