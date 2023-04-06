#include <exception>

#include "matrix.hpp"
#include "maze.hpp"

Maze::Maze(
  const Matrix<cell_t>& m
) :
  _end_pos({0, 0}),
  _start_pos({0, 0}),
  _config(Matrix<cell_t>::full(m.rows, m.cols, Cell::dead))
{
  bool has_start = false;
  bool has_end = false;

  // loop over cells to check and search for start/end cells
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const auto idx = i*cols + j;
      switch (m[idx]) {
        case Cell::dead:
          break;
        case Cell::live:
          set_cell(idx);
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

      // set the border bit
      if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        _config[idx] |= 2;
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

  const auto i = idx/cols;
  const auto j = idx%cols;

  if (_config[idx] & 2) {
    if (i > 0) {
      _config[idx-cols] += 4;
      if (j < cols-1) {
        _config[idx-cols+1] += 4;
      }
      if (j > 0) {
        _config[idx-cols-1] += 4;
      }
    }

    if (j > 0) {
      _config[idx-1] += 4;
      if (i < rows-1) {
        _config[idx+cols-1] += 4;
      }
    }

    if (i < rows-1) {
      _config[idx+cols] += 4;
      if (j < cols-1) {
        _config[idx+cols+1] += 4;
      }
    }

    if (j < cols-1) {
      _config[idx+1] += 4;
    }
  } else {
    _config[idx-cols-1] += 4;
    _config[idx-cols] += 4;
    _config[idx-cols+1] += 4;
    _config[idx-1] += 4;
    _config[idx+1] += 4;
    _config[idx+cols-1] += 4;
    _config[idx+cols] += 4;
    _config[idx+cols+1] += 4;
  }
}

void
Maze::clear_cell(
  const std::size_t idx
)
{
  _config[idx] ^= Cell::live;

  const auto i = idx/cols;
  const auto j = idx%cols;

  if (_config[idx] & 2) {
    if (i > 0) {
      _config[idx-cols] -= 4;
      if (j < cols-1) {
        _config[idx-cols+1] -= 4;
      }
      if (j > 0) {
        _config[idx-cols-1] -= 4;
      }
    }

    if (j > 0) {
      _config[idx-1] -= 4;
      if (i < rows-1) {
        _config[idx+cols-1] -= 4;
      }
    }

    if (i < rows-1) {
      _config[idx+cols] -= 4;
      if (j < cols-1) {
        _config[idx+cols+1] -= 4;
      }
    }

    if (j < cols-1) {
      _config[idx+1] -= 4;
    }
  } else {
    _config[idx-cols-1] -= 4;
    _config[idx-cols] -= 4;
    _config[idx-cols+1] -= 4;
    _config[idx-1] -= 4;
    _config[idx+1] -= 4;
    _config[idx+cols-1] -= 4;
    _config[idx+cols] -= 4;
    _config[idx+cols+1] -= 4;
  }
}

void
Maze::set_cell(
  const std::size_t i,
  const std::size_t j
)
{
  set_cell(i*cols + j);
}

void
Maze::clear_cell(
  const std::size_t i,
  const std::size_t j
)
{
  clear_cell(i*cols + j);
} 

Maze&
Maze::evolve()
{
  auto m = Matrix<cell_t>(config);

  for (std::size_t idx = 0; idx < rows*cols; ++idx) {
    const auto count = m[idx] / 4;

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

  // restore start/end cells in case they have changed
  _config[start_pos] = Cell::dead;
  _config[end_pos] = Cell::dead;

  return *this;
}
