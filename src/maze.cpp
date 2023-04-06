#include <exception>

#include "matrix.hpp"
#include "maze.hpp"

Maze::Maze(
  const Matrix<uint>& m
) :
  _rows(m.rows + 2),
  _cols(m.cols + 2),
  _end_pos({0, 0}),
  _start_pos({0, 0}),
  _config(Matrix<uint>::full(m.rows + 2, m.cols + 2, Cell::dead))
{
  bool has_start = false;
  bool has_end = false;

  // loop over cells to check and search for start/end cells
  for (std::size_t i = 1; i < rows-1; ++i) {
    for (std::size_t j = 1; j < cols-1; ++j) {
      const auto& state = m[{i-1, j-1}];
      switch (state) {
        case Cell::dead:
          break;
        case Cell::live:
          set_cell(i*cols + j);
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
  const std::size_t idx
)
{
  _config[idx] ^= Cell::live;

  _config[idx-cols] += 2;
  _config[idx-cols-1] += 2;
  _config[idx-cols+1] += 2;
  _config[idx-1] += 2;
  _config[idx+1] += 2;
  _config[idx+cols] += 2;
  _config[idx+cols-1] += 2;
  _config[idx+cols+1] += 2;
}

void
Maze::clear_cell(
  const std::size_t idx
)
{
  _config[idx] ^= Cell::live;

  _config[idx-cols-1] -= 2;
  _config[idx-cols] -= 2;
  _config[idx-cols+1] -= 2;
  _config[idx-1] -= 2;
  _config[idx+1] -= 2;
  _config[idx+cols-1] -= 2;
  _config[idx+cols] -= 2;
  _config[idx+cols+1] -= 2;
}

Maze&
Maze::evolve()
{
  auto m = Matrix<uint>(config);
  
  for (std::size_t i = 1; i < rows-1; ++i) {
    for (std::size_t j = 1; j < cols-1; ++j) {
      const auto idx = i*cols + j;
      const auto count = m[idx] / 2;

      if (m[idx] & Cell::live) {
        if ((count < 4 || count > 6)) {
          clear_cell(idx);
        }
      } else {
        if (1 < count && count < 4) {
          set_cell(idx);
        }
      }
    }
  }

  // restore start/end cells in case they have changed
  _config[start_pos] = Cell::dead;
  _config[end_pos] = Cell::dead;

  return *this;
}
