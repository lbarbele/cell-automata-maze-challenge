#ifndef _has_maze_hpp_
#define _has_maze_hpp_

#include "matrix.hpp"
#include "position.hpp"

using cell_t = std::uint16_t;

class Maze {
private:
  Position _end_pos;
  Position _start_pos;
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
    static const cell_t start = 3;
    static const cell_t end = 4;
  };

public:

  const Matrix<cell_t>& config = _config;
  const std::size_t& rows = config.rows;
  const std::size_t& cols = config.cols;
  const Position& end_pos = _end_pos;
  const Position& start_pos = _start_pos;

  Maze(const Matrix<cell_t>& m);

  Maze& evolve();
};

#endif // _has_maze_hpp_