#include <pybind11/pybind11.h>
#include "fundarg.hpp"
#include "solid_earth_tide.hpp"
#include "earth_rotation.hpp"
#include "planets.hpp"

namespace py = pybind11;

/* The Python module will be called "pyiers10" */
PYBIND11_MODULE(pyiers10, m) {
    m.doc() = "Python bindings for the iers library";

    /* Expose the MyClass class */
    py::class_<dso::SolidEarthTide>(m, "SolidEarthTide")
      .def("displacement", &dso::SolidEarthTide::displacement, "Compute site displacement due to Solid Earth tide according to IERS 2010.");
}
