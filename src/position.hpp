#ifndef _has_position_hpp_
#define _has_position_hpp_

#include <algorithm>
#include <vector>

struct Position {
  std::size_t x;
  std::size_t y;

  const bool operator==(const Position& other) const
  {return x == other.x && y == other.y;}

  std::size_t distance(const Position& other) const
  {return std::max(x, other.x) - std::min(x, other.x) + std::max(y, other.y) - std::min(y, other.y);}

  std::vector<Position> get_neighbours(
    const std::size_t rows,
    const std::size_t cols
  ) const {
    std::vector<Position> v;
    v.reserve(4);
    if (x > 0) v.push_back({x-1, y});
    if (x < rows-1) v.push_back({x+1, y});
    if (y > 0) v.push_back({x, y-1});
    if (x < cols-1) v.push_back({x, y+1});
    return v;
  }
};

// position printer function
template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const Position& p) {
  return os << "(" << p.x << ", " << p.y << ")";
}

#endif // _has_position_hpp_
