#include <exception>

#include "matrix.hpp"
#include "maze.hpp"

Maze::Maze(
  const Matrix<uint>& m
) :
  _end_pos({0, 0}),
  _start_pos({0, 0}),
  _config(Matrix<uint>::full(m.rows, m.cols, 0))
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
          set_cell(i, j);
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

void
Maze::set_cell(
  const std::size_t i,
  const std::size_t j
)
{
  _config[{i, j}] = Cell::live;
}

void
Maze::clear_cell(
  const std::size_t i,
  const std::size_t j
)
{
  _config[{i, j}] = Cell::dead;
}

uint&
Maze::cell_state(
  const std::size_t i,
  const std::size_t j
)
{
  return _config[{i, j}];
}

Maze
Maze::evolve()
{
  auto m = Matrix<uint>(config);

  for (std::size_t idx = 0; idx < rows*cols; ++idx) {
    const auto i = idx/cols;
    const auto j = idx%cols;

    uint count = 0;

    if (i > 0) {
      if (j > 0) {
        count += m[idx-1];
        count += m[idx-cols];
        count += m[idx-cols-1];
      }
      if (j < cols-1) {
        count += m[idx-cols+1];
      }
    }

    if (i < rows-1) {
      if (j > 0) {
        count += m[idx+cols-1];
      }
      if (j < cols-1) {
        count += m[idx+1];
        count += m[idx+cols];
        count += m[idx+cols+1];
      }
    }

    if (m[idx] == Cell::live) {
      if (3 >= count || count >= 7) {
        _config[idx] = Cell::dead;
      }
    } else {
      if (1 < count && count < 4) {
        _config[idx] = Cell::live;
      }
    }
  }

  // restore start/end cells in case they have changed
  _config[start_pos] = Cell::dead;
  _config[end_pos] = Cell::dead;

  return *this;
}
