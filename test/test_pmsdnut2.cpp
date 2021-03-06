#include "iers2010.hpp"
#include "test_help.hpp"
#include <cassert>

constexpr double _alg_accuracy_ = 1e-6;

int main() {

  // testing pmsdnut2
  std::cout << "----------------------------------------\n";
  std::cout << "> pmsdnut2\n";
  std::cout << "----------------------------------------\n";

  const double dx_ref = 24.83144238273364834e0;
  const double dy_ref = -14.09240692041837661e0;
  double dx, dy;
  iers2010::pmsdnut2(54335e0, dx, dy);

#ifdef STRICT_TEST
  assert(approxEqual(dx, dx_ref));
  assert(approxEqual(dy, dy_ref));
#else
  printf("\ndx= %12.6e microarcseconds", std::abs(dx - dx_ref));
  printf("\ndy= %12.6e microarcseconds", std::abs(dy - dy_ref));
  assert(std::abs(dx - dx_ref) < _alg_accuracy_ &&
         std::abs(dy - dy_ref) < _alg_accuracy_);
#endif

  return 0;
}
