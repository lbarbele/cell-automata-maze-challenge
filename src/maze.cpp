#include <exception>

#include "matrix.hpp"
#include "maze.hpp"

Maze::Maze(
  const Matrix<uint>& m
) :
  _config(Matrix<uint>::full(m.rows, m.cols, 0)),
  _end_pos({0, 0}),
  _start_pos({0, 0})
{
  bool has_start = false;
  bool has_end = false;

  // loop over cells to check them and search for the start/end cells
  for (std::size_t i = 0; i < m.rows*m.cols; ++i) {
    const auto& cell = m[i];
    switch (cell) {
      case Cell::dead:
        break;
      case Cell::live:
        _config[i] = cell;
        break;
      case Cell::start:
        if (!has_start) {
          _start_pos = {i/cols, i%cols};
          has_start = true;
        } else {
          throw std::runtime_error("matrix has multiple starting cells");
        }
        break;
      case Cell::end:
        if (!has_end) {
          _end_pos = {i/cols, i%cols};
          has_end = true;
        } else {
          throw std::runtime_error("matrix has multiple ending cells");
        }
        break;
      default:
        throw std::runtime_error("invalid cell type " + std::to_string(cell));
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

  // central cells
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      count = 0;
      if (i > 0 && j > 0) count += m[{i-1, j-1}];
      if (i > 0) count += m[{i-1, j}];
      if (i > 0 && j < cols - 1) count += m[{i-1, j+1}];
      if (j > 0) count += m[{  i, j-1}];
      if (j < cols - 1) count += m[{  i, j+1}];
      if (i < rows - 1 && j > 0) count += m[{i+1, j-1}];
      if (i < rows - 1) count += m[{i+1, j}];
      if (i < rows - 1 && j < cols - 1) count += m[{i+1, j+1}];

      const Position pos = {i, j};

      auto& state = _config[pos];

      if ((state == Cell::live) && (3 >= count || count >= 7)) {
        state = Cell::dead;
      } else if ((state == Cell::dead) && (1 < count && count < 4)) {
        state = Cell::live;
      }
    }
  }

  // restore start/end cells
  _config[start_pos] = Cell::start;
  _config[end_pos] = Cell::end;

  return *this;
}
