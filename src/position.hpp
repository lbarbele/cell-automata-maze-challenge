#ifndef _has_position_hpp_
#define _has_position_hpp_

#include <algorithm>
#include <vector>

namespace utl {

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

    // function to get neighbour positions on a grid

    std::vector<position>
    get_neighbours(
      const std::size_t rows,
      const std::size_t cols
    ) const {
      std::vector<position> v;
      v.reserve(4);
      if (x > 0)      v.push_back({x-1, y});
      if (x < rows-1) v.push_back({x+1, y});
      if (y > 0)      v.push_back({x, y-1});
      if (x < cols-1) v.push_back({x, y+1});
      return v;
    }
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

} // utl

#endif // _has_position_hpp_
