#ifndef _has_position_hpp_
#define _has_position_hpp_

#include <algorithm>
#include <list>
#include <vector>

namespace utl {

  struct position;

  // aliases

  using position_list = std::list<position>;

  // actual class definition

  struct position {
    std::size_t x;
    std::size_t y;

    // compute distance to another position

    std::size_t
    distance(
      const position& other
    ) const
    {
      const auto dx = std::max(x, other.x) - std::min(x, other.x);
      const auto dy = std::max(y, other.y) - std::min(y, other.y);
      return dx + dy;
    }

    // operators

    const bool operator==(const position& other) const
    {return x == other.x && y == other.y;}

  };

  // position printer

  template <class CharT, class Traits>
  std::basic_ostream<CharT, Traits>&
  operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const position& p
  ) {
    return os << "(" << p.x << ", " << p.y << ")";
  }

  // position_list printer

  template <class CharT, class Traits>
  std::basic_ostream<CharT, Traits>&
  operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const position_list& l
  ) {
    for (const auto & p : l) os << p << std::endl;
    return os;
  }


} // utl

#endif // _has_position_hpp_
