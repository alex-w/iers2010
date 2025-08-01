#include "earth_rotation.hpp"
#include "fundarg.hpp"
#include "planets.hpp"
#include "solid_earth_tide.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

/* The Python module will be called "pyiers10" */
PYBIND11_MODULE(pyiers10, m) {
  m.doc() = "Python bindings for the iers library";

  /* Expose the MyClass class */
  py::class_<dso::SolidEarthTide>(m, "SolidEarthTide")
      .def(py::init<double, double, double, double>(),
           py::arg("GMearth") = ::iers2010::GMe,
           py::arg("Rearth") = ::iers2010::Re,
           py::arg("GMsun") = 1.32712442076e20,
           py::arg("GMmoon") = 0.49028010560e13,
           "Create a SolidEarthTide object with optional gravitational "
           "parameters")
      .def(
          "displacement", &dso::SolidEarthTide::displacement,
          "Compute site displacement due to Solid Earth tide according to IERS "
          "2010.");
}
