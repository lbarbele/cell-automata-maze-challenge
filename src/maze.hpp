#ifndef _has_maze_hpp_
#define _has_maze_hpp_

#include <filesystem>

#include "matrix.hpp"
#include "position.hpp"

using cell_t = std::uint16_t;

class Maze {
private:
  Matrix<cell_t> _config;

  void set_cell(const std::size_t idx);
  void set_cell(const std::size_t i, const std::size_t j) {set_cell(i*cols + j);}
  void set_cell(const Position& p) {set_cell(p.x, p.y);}
  void clear_cell(const std::size_t idx);
  void clear_cell(const std::size_t i, const std::size_t j) {clear_cell(i*cols + j);}
  void clear_cell(const Position& p) {clear_cell(p.x, p.y);}  
  
  struct Cell {
    static const cell_t dead = 0;
    static const cell_t live = 1;
  };

public:

  const Matrix<cell_t>& config = _config;
  const std::size_t& rows = config.rows;
  const std::size_t& cols = config.cols;

  Maze(const Maze& other);
  Maze(const Matrix<cell_t>& m, const bool ignore_bad = false);

  static Maze from_file(std::filesystem::path path, const bool ignore_bad = false)
  {return Maze(Matrix<cell_t>::from_file(path), ignore_bad);}

  Maze& evolve(const uint generations = 1);
};

// maze printer
template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const Maze& m) {
  for (std::size_t i = 0; i < m.rows*m.cols; ++i) {
    os << std::setw(2) << m.config[i]%2;
    if ((i+1)%m.cols == 0)
      os << std::endl;
  }
  return os;
}

#endif // _has_maze_hpp_