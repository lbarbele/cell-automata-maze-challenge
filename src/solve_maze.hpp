#ifndef _has_solve_maze_hpp_
#define _has_solve_maze_hpp_

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include "cell_maze.hpp"
#include "matrix.hpp"
#include "path.hpp"
#include "position.hpp"

namespace utl {

  std::string
  solution_to_string(
    const position_list& solution
  )
  {
    // convert list to vectors
    std::vector<position> v(solution.begin(), solution.end());

    std::stringstream ss;

    // walk through the solution to convert into step strings
    for (std::size_t i = 1; i < v.size(); ++i) {
      const auto& cur = v[i];
      const auto& prv = v[i-1];

      if (cur.x == prv.x + 1)
        ss << "D ";
      else if (cur.x + 1 == prv.x)
        ss << "U ";
      else if (cur.y == prv.y + 1)
        ss << "R ";
      else if (cur.y + 1 == prv.y)
        ss << "L ";
      else
        throw std::runtime_error("invalid solution");
    }

    // convert to string
    auto str = ss.str();

    // remove training space
    if (str.back() == ' ')
      str.pop_back();

    return str;
  }

  template <class T>
  position_list
  solve_maze(
    const utl::cell_maze<T>& input_maze,
    const utl::position& start_pos,
    const utl::position& end_pos,
    const uint max_steps,
    const uint lives = 1,
    const std::size_t drop_distance = 100,
    const bool verbose = true
  )
  {
    using path_t = utl::path<utl::position>;

    // get a copy of the input maze
    auto maze = input_maze;

    // this is the list of all paths up to the given step
    std::list<path_t::ptr_t> all_paths = {path_t::create(start_pos)};

    // walk up to max_steps steps
    for (uint istep = 1; istep < max_steps; ++istep) {

      // update the maze for the current step
      maze.evolve();

      // restore start/end cells, if necessary
      maze.clear_cell(start_pos);
      maze.clear_cell(end_pos);

      // list with the paths that will be available in the next step
      std::list<path_t::ptr_t> next_all_paths;

      // keep track of the positions already reached in the current step
      auto occupation = utl::matrix<uint>::full(maze.rows(), maze.cols(), 0);

      // keep track of the shortest distance to the end cell
      std::size_t shortest_distance = maze.rows() + maze.cols();

      // loop over all paths with the current number of steps
      for (auto& cur_path : all_paths) {

        // loop over all possible moves
        const auto moves = maze.get_neighbours(cur_path->get_position());
        for (const auto& pos : moves) {

          // if the current path leads to the end cell, we are done
          if (pos == end_pos) {
            auto solution = cur_path->walk(pos);
            return solution->get_position_list();
          }

          const auto dist = pos.distance(end_pos);
          shortest_distance = std::min(dist, shortest_distance);

          // if the cell is already occupied, skip it
          if (occupation[pos]) {
            continue;
          }

          if (maze.is_alive(pos)) {
            // if position leads to a live cell and lives > 1, step on it
            if (cur_path->get_lives() > 1) {
              auto move = cur_path->walk(pos, cur_path->get_lives() - 1);
              next_all_paths.push_back(move);
              ++occupation[pos];
            }
          } else {
            // paths leads to a dead cell
            auto move = cur_path->walk(pos);
            next_all_paths.push_back(move);
            ++occupation[pos];
          }
        }

        // drop the current path, in case it leads to nowhere
        if (cur_path->get_next().empty()) {
          cur_path->drop();
        }
      }

      // update list of paths
      all_paths.clear();

      for (auto& path : next_all_paths) {
        if (path->get_position().distance(end_pos) - shortest_distance < drop_distance) {
          all_paths.push_back(path);
        } else {
          path->drop();
        }
      }

      if (verbose) {
        std::cout
          << "step " << istep
          << ", paths " << all_paths.size()
          << ", shortest distance to finish " << shortest_distance
          << std::endl;
      }
    }

    return position_list{};
  }

} // utl

#endif // _has_solve_maze_hpp_