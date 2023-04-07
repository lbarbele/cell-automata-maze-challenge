#ifndef _has_clock_hpp_
#define _has_clock_hpp_

#include <chrono>
#include <iostream>

namespace utl {

  template <class U = std::chrono::milliseconds>
  class timer {
  private:
    decltype(std::chrono::system_clock::now()) t;

  public:
    using unit_t = U;

    timer() {restart();}

    timer&
    restart()
    {
      t = std::chrono::system_clock::now();
      return *this;
    }

    auto get() const
    {return std::chrono::duration_cast<unit_t>(timer().t - t);}
  };


  template <class CharT, class Traits, class U>
  std::basic_ostream<CharT, Traits>&
  operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const timer<U>& c
  ) {
    return os << c.get();
  }

} // ::utl

#endif // _has_clock_hpp_