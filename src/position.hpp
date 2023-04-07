#ifndef _has_position_hpp_
#define _has_position_hpp_

#include <vector>

struct Position {
  std::size_t x;
  std::size_t y;

  const bool operator==(const Position& other) const {
    return x == other.x && y == other.y;
  }
};

// position printer function
template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const Position& p) {
  return os << "(" << p.x << ", " << p.y << ")";
}

#endif // _has_position_hpp_
