#include "iers2010.hpp"
#include "test_help.hpp"
#include <cassert>

int main() {

  // testing fundarg
  std::cout << "----------------------------------------\n";
  std::cout << "> fundarg\n";
  std::cout << "----------------------------------------\n";
  double fargs[5],fargs_ref[]={2.291187512612069099e0, 6.212931111003726414e0, 3.658025792050572989e0, 4.554139562402433228e0,  -0.5167379217231804489e0};
  iers2010::fundarg(0.07995893223819302e0, fargs);
  for (int i = 0; i < 5; ++i) {
#ifdef STRICT_TEST
    assert(approxEqual(fargs[i], fargs_ref[i]));
#else
    if (!approxEqual(fargs[i], fargs_ref[i])) {
      printf("\nargs[%1d] = %e", i, std::abs(fargs[i]- fargs_ref[i]));
    }
    assert(std::abs(fargs[i]- fargs_ref[i])<1e-11);
#endif
  }
  return 0;
}
