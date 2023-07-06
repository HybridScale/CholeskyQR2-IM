function(add_imported_library library headers)
  add_library(NCCL::NCCL UNKNOWN IMPORTED)
  set_target_properties(NCCL::NCCL PROPERTIES
    IMPORTED_LOCATION ${library}
    INTERFACE_INCLUDE_DIRECTORIES ${headers}
  )
  set(NCCL_FOUND 1 CACHE INTERNAL "NCCL found" FORCE)
  set(NCCL_LIBRARIES ${library}
      CACHE STRING "Path to nccl library" FORCE)
  set(NCCL_INCLUDES ${headers}
      CACHE STRING "Path to nccl headers" FORCE)
  mark_as_advanced(FORCE NCCL_LIBRARIES)
  mark_as_advanced(FORCE NCCL_INCLUDES)
endfunction()

if (NCCL_LIBRARIES AND NCCL_INCLUDES)
  add_imported_library(${NCCL_LIBRARIES} ${NCCL_INCLUDES})
  return()
endif()

find_library(NCCL_LIBRARY_PATH NAMES libnccl nccl)
find_path(NCCL_HEADER_PATH NAMES nccl.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NCCL DEFAULT_MSG NCCL_LIBRARY_PATH NCCL_HEADER_PATH
)
if (NCCL_FOUND)
  add_imported_library(
    "${NCCL_LIBRARY_PATH}"
    "${NCCL_HEADER_PATH}"
  )
endif()