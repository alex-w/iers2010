target_sources(iers
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src/lib/solid_earth_tides/iers2010_displacement_step1.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/solid_earth_tides/iers2010_displacement_step2.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/solid_earth_tides/iers2010_geopotential_step1.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/solid_earth_tides/iers2010_geopotential_step2.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/solid_earth_tides/iers2010_site_displacement.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/solid_earth_tides/iers2010_stokes_coeffs.cpp
)
