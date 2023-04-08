#ifndef _has_cell_maze_hpp_
#define _has_cell_maze_hpp_

#include <concepts>
#include <filesystem>

#include "matrix.hpp"
#include "position.hpp"

namespace utl {

  template <std::unsigned_integral CellT>
  class cell_maze {
  public: // type definitions
    using cell_t = CellT;

  private: // data fields
    matrix<cell_t> _config;
    cell_t _rule_underpop  = 3; // cell count representing underpopulation
    cell_t _rule_overpop   = 6; // cell count representing overpopulation
    cell_t _rule_repro_min = 2; // minimum cell count for reproduction
    cell_t _rule_repro_max = 4; // maximum cell count for reproduction

    static const cell_t _dead_cell     = 0b000;
    static const cell_t _live_cell     = 0b001;
    static const cell_t _border_bit    = 0b010;
    static const cell_t _count_padding = 0b100;

  private: // methods

    // rules checker

    bool
    _check_rules()
    {
      return _rule_repro_max >= _rule_repro_min && _rule_underpop < _rule_overpop;
    }

    // apply propagation rules to cell

    void
    _apply_rules(
      const std::size_t idx,
      const cell_t neighbour_count
    )
    {
      if (is_alive(idx)) {
        if (neighbour_count <= _rule_underpop || neighbour_count >= _rule_overpop)
          clear_cell(idx);
      } else {
        if (_rule_repro_min <= neighbour_count && neighbour_count <= _rule_repro_max)
          set_cell(idx);
      }
    }

    // cell updater

    void
    _update_cell(
      const std::size_t idx,
      const cell_t state
    )
    {
      // get current cell state
      auto& cur_state = _config[idx];

      // do nothing if required state is the same as the current state
      if ((cur_state & _live_cell) == state) {
        return;
      }

      // switch cell state
      _config[idx] ^= _live_cell;

      // determine the count change value
      const auto dcount = state ? _count_padding : -_count_padding;

      // loop over neighbouring cells to update their neighbour count
      // for cells on the border, take care of not trespassing array
      // limits. if the border bit is disabled, operate on all 8
      // adjacent cells
      if (is_border(idx)) {
        const auto i = idx/cols();
        const auto j = idx%cols();

        if (i > 0) {
          _config[idx-cols()] += dcount;
          if (j < cols()-1)
            _config[idx-cols()+1] += dcount;
          if (j > 0)
            _config[idx-cols()-1] += dcount;
        }

        if (j > 0) {
          _config[idx-1] += dcount;
          if (i < rows()-1)
            _config[idx+cols()-1] += dcount;
        }

        if (i < rows()-1) {
          _config[idx+cols()] += dcount;
          if (j < cols()-1)
            _config[idx+cols()+1] += dcount;
        }

        if (j < cols()-1)
          _config[idx+1] += dcount;
      } else {
        _config[idx-cols()-1] += dcount;
        _config[idx-cols()]   += dcount;
        _config[idx-cols()+1] += dcount;
        _config[idx-1]        += dcount;
        _config[idx+1]        += dcount;
        _config[idx+cols()-1] += dcount;
        _config[idx+cols()]   += dcount;
        _config[idx+cols()+1] += dcount;
      }
    }

    // enable border bit on all border cells

    void
    _set_border()
    {
      for (std::size_t i = 0; i < rows(); ++i) {
        _config[{i, 0}]        |= _border_bit;
        _config[{i, cols()-1}] |= _border_bit;
      }

      for (std::size_t j = 1; j < cols() - 1; ++j) {
        _config[{0, j}]        |= _border_bit;
        _config[{rows()-1, j}] |= _border_bit;
      }
    }

  public: // methods

    // construct empty maze

    cell_maze(
      const std::size_t m,
      const std::size_t n
    ) :
      _config(matrix<cell_t>::full(m, n, _dead_cell))
    {
      _set_border();
    }

    // construct from matrix

    cell_maze(
      const matrix<cell_t>& m,
      const bool ignore_bad = false
    ) :
      _config(matrix<cell_t>::full(m.rows(), m.cols(), _dead_cell))
    {
      _set_border();

      // loop over matrix elements to get the initial configuration
      for (std::size_t idx = 0; idx < size(); ++idx) {
        switch (m[idx]) {
        case _dead_cell:
          break;
        case _live_cell:
          set_cell(idx);
          break;
        default:
          if (!ignore_bad) {
            throw std::runtime_error("invalid cell type " + std::to_string(m[idx]));
          }
        }
      }
    }

    // read maze from file

    static
    cell_maze
    from_file(
      std::filesystem::path path,
      const bool ignore_bad = false
    )
    {
      return cell_maze(matrix<cell_t>::from_file(path), ignore_bad);
    }

    // retrieve cell state

    auto operator[](const position& p) const {return config()[p] & _live_cell;}
    auto operator[](const std::size_t idx) const {return config()[idx] & _live_cell;}

    // retrive matrix of cell states

    auto states() const {return config()&_live_cell;}

    // retrive matrix of neighbour counts

    auto neighbour_count() const {return config()/_count_padding;}

    // cell setter

    void set_cell(const std::size_t idx) {_update_cell(idx, _live_cell);}
    void set_cell(const std::size_t i, const std::size_t j) {set_cell(i*cols() + j);}
    void set_cell(const position& p) {set_cell(p.x, p.y);}

    // cell clearer

    void clear_cell(const std::size_t idx) {_update_cell(idx, _dead_cell);}
    void clear_cell(const std::size_t i, const std::size_t j) {clear_cell(i*cols() + j);}
    void clear_cell(const position& p) {clear_cell(p.x, p.y);}

    // check if position is on border

    bool is_border(const position& p) {return _config[p] & _border_bit;}
    bool is_border(const std::size_t idx) {return _config[idx] & _border_bit;}

    // check if cell is alive/dead

    bool is_alive(const position& p) {return _config[p] & _live_cell;}
    bool is_alive(const std::size_t idx) {return _config[idx] & _live_cell;}

    // rule setters

    void set_overpopulation_rule(const cell_t value) {
      _rule_overpop = value;
    }

    void set_underpopulation_rule(const cell_t value) {
      _rule_underpop = value;
    }

    void set_reproduction_rules(
      const cell_t min,
      const cell_t max
    ) {
      if (max < min) {
        throw std::runtime_error("bad reproduction rule: max < min");
      }

      _rule_repro_min = min;
      _rule_repro_max = max;
    }

    // get neighbour positions of given cell

    position_list
    get_neighbours(
      const position& p
    )
    {
      const auto& [i, j] = p;

      if (is_border(p)) {
        position_list l;
        if (i > 0)        l.emplace_back(i-1, j);
        if (j > 0)        l.emplace_back(i, j-1);
        if (i < rows()-1) l.emplace_back(i+1, j);
        if (j < cols()-1) l.emplace_back(i, j+1);
        return l;
      } else {
        return {{i-1, j}, {i+1, j}, {i, j-1}, {i, j+1}};
      }
    }

    // evolve maze by n generations

    cell_maze&
    evolve(
      const uint generations = 1
    )
    {
      // before applying the propagation rules, check them
      if (!_check_rules()) {
        throw std::runtime_error("invalid propagation rules!");
      }

      // loop over generations
      for (uint i = 0; i < generations; ++i) {
        // get a matrix representing the cell counts
        const auto count = config()/_count_padding;

        // iterate over all cells, applying the propagation rules to all
        for (std::size_t idx = 0; idx < size(); ++idx) {
          _apply_rules(idx, count[idx]);
        }
      }

      return *this;
    }

    // maze dimensions

    auto cols() const {return config().cols();}
    auto rows() const {return config().rows();}
    auto size() const {return config().size();}

    // access to the configuration matrix

    const auto& config() const {return _config;}

  };

  // maze printer
  template <class CharT, class Traits, class C>
  std::basic_ostream<CharT, Traits>&
  operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const cell_maze<C>& m
  ) {
    return os << (m.config()&1);
  }

}

#endif // _has_cell_maze_hpp_