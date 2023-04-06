#ifndef _has_maze_hpp_
#define _has_maze_hpp_

#include <chrono>

#include "matrix.hpp"
#include "position.hpp"

class Maze {
private:
  std::size_t _rows;
  std::size_t _cols;
  Position _end_pos;
  Position _start_pos;
  Matrix<uint> _config;

  void set_cell(const std::size_t idx);
  void clear_cell(const std::size_t idx);

public:
  struct Cell {
    static const uint dead = 0;
    static const uint live = 1;
    static const uint start = 3;
    static const uint end = 4;
  };

  const Matrix<uint>& config = _config;
  const std::size_t& rows = _rows;
  const std::size_t& cols = _cols;
  const Position& end_pos = _end_pos;
  const Position& start_pos = _start_pos;

  Maze(const Matrix<uint>& m);

  Maze& evolve();
};

#endif // _has_maze_hpp_