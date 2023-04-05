#include <exception>

#include "matrix.hpp"
#include "maze.hpp"

Maze::Maze(
  const Matrix<uint>& m
) :
  _rows(m.rows),
  _cols(m.cols),
  _end_pos({0, 0}),
  _start_pos({0, 0}),
  _config(Matrix<uint>::full(m.rows + 2, m.cols + 2, 0))
{
  bool has_start = false;
  bool has_end = false;

  // loop over cells to check and search for start/end cells
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      switch (m[{i, j}]) {
        case Cell::dead:
          break;
        case Cell::live:
          _config[{i+1, j+1}] = Cell::live;
          break;
        case Cell::start:
          if (!has_start) {
            _start_pos = {i, j};
            has_start = true;
          } else {
            throw std::runtime_error("matrix has multiple starting cells");
          }
          break;
        case Cell::end:
          if (!has_end) {
            _end_pos = {i, j};
            has_end = true;
          } else {
            throw std::runtime_error("matrix has multiple ending cells");
          }
          break;
        default:
          throw std::runtime_error("invalid cell type " + std::to_string(m[{i, j}]));
      }
    }
  }

  // check start/end cells
  if (!has_start) {
    throw std::runtime_error("matrix does not contain a starting cell");
  }

  if (!has_end) {
    throw std::runtime_error("matrix does not contain an ending cell");
  }
}

Maze
Maze::evolve()
{
  auto m = Matrix<uint>(config);

  uint count = 0;

  for (std::size_t i = 1; i <= rows; ++i) {
    for (std::size_t j = 1; j <= cols; ++j) {
      count =
        m[{i-1, j-1}] + m[{i-1, j}] + m[{i-1, j+1}] +
        m[{  i, j-1}] +               m[{  i, j+1}] + 
        m[{i+1, j-1}] + m[{i+1, j}] + m[{i+1, j+1}];

      auto& state = _config[{i, j}];

      if ((state == Cell::live) && (3 >= count || count >= 7)) {
        state = Cell::dead;
      } else if ((state == Cell::dead) && (1 < count && count < 4)) {
        state = Cell::live;
      }
    }
  }

  // restore start/end cells in case they have changed
  _config[{start_pos.x + 1, start_pos.y + 1}] = Cell::dead;
  _config[{end_pos.x + 1, end_pos.y + 1}] = Cell::dead;

  return *this;
}
