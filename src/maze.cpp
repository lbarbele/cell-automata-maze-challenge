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
  _config[{i+1, j+1}] = Cell::live;
}

void
Maze::clear_cell(
  const std::size_t i,
  const std::size_t j
)
{
  _config[{i+1, j+1}] = Cell::dead;
}

uint&
Maze::cell_state(
  const std::size_t i,
  const std::size_t j
)
{
  return _config[{i+1, j+1}];
}

Maze
Maze::evolve()
{
  auto m = Matrix<uint>(config);

  uint count = 0;

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      count =
        m[{  i, j}] + m[{  i, j+1}] + m[{  i, j+2}] +
        m[{i+1, j}] +                 m[{i+1, j+2}] + 
        m[{i+2, j}] + m[{i+2, j+1}] + m[{i+2, j+2}];

      if (cell_state(i, j) == Cell::live) {
        if (3 >= count || count >= 7) {
          clear_cell(i, j);
        }
      } else {
        if (1 < count && count < 4) {
          set_cell(i, j);
        }
      }
    }
  }

  // restore start/end cells in case they have changed
  _config[{start_pos.x + 1, start_pos.y + 1}] = Cell::dead;
  _config[{end_pos.x + 1, end_pos.y + 1}] = Cell::dead;

  return *this;
}
