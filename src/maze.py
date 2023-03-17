import random

import matplotlib.pyplot as plt
import numpy as np

WHITE = 0
GREEN = 1
START = 2
FINISH = 3

class Maze():
  def __init__(self, shape: tuple[int, int], start: tuple[int, int] = None, finish: tuple[int, int] = None, greens = list[tuple[int, int]]):
    self.shape = shape
    self.rows = shape[0]
    self.cols = shape[1]

    self.start = (0, 0) if start is None else start
    self.finish = (-1, -1) if finish is None else finish

    self.maze = np.full(shape, WHITE)
    self.maze[self.start] = START
    self.maze[self.finish] = FINISH
    for ij in greens:
      self.maze[ij] = GREEN

  def clone(self):
    return Maze(self.shape, self.start, self.finish, self.get_cell_indices(GREEN))

  def get_cell_indices(self, cell_type):
    return [pos for pos, c in np.ndenumerate(self.maze) if c == cell_type]

  def draw(self, position: tuple[int, int] = None):
    plt.matshow(self.maze, cmap = 'tab10')
    if not (position is None):
      plt.plot(position[1], position[0], 'kD', ms = 20)
      moves = np.array(self.get_valid_moves(position))
      if len(moves) > 0:
        plt.plot(moves[:, 1], moves[:, 0], 'k+')
    plt.show()

  def get_neighbours(self, position: tuple[int, int], include_diag: bool = True):
    i, j = position

    if i < 0: i = self.rows + i
    if j < 0: j = self.cols + j

    # vertical neighbours
    nb = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]

    # diagonal neighbours
    if include_diag:
      nb = nb + [(i+1, j+1), (i+1, j-1), (i-1, j-1), (i-1, j+1)]

    return [(k, l) for k, l in nb if (k>=0 and l>=0 and k<self.rows and l<self.cols)]
  
  def count_green_neighbours(self, position: tuple[int, int]):
    count = 0

    for ij in self.get_neighbours(position):
      if self.maze[ij] == GREEN:
        count += 1
    
    return count
  
  def get_mutation(self, position: tuple[int, int]):
    type = self.maze[position]

    if type == START or type == FINISH:
      return type
    
    ngreen = self.count_green_neighbours(position)
    if type == WHITE and (ngreen == 2 or ngreen == 3):
      return GREEN
    elif type == GREEN and ngreen >= 4:
      return GREEN
    else:
      return WHITE
  
  def evolve(self, overwrite = True):
    other = np.full(self.maze.shape, WHITE)

    for pos in np.ndindex(self.shape):
      other[pos] = self.get_mutation(pos)

    if overwrite:
      self.maze = other

    return other

  def get_valid_moves(self, position: tuple[int, int] = None):
    if position is None:
      position = self.start

    if self.maze[position] == GREEN:
      return []
    
    nb = self.get_neighbours(position, include_diag = False)
    valid_moves = [ij for ij in nb if self.get_mutation(ij) != GREEN]
    return valid_moves
  
  def random_walk(self, position: tuple[int, int], evolve = False):
    if position is None:
      return None
    
    moves = self.get_valid_moves(position)

    if len(moves) == 0:
      return None
    
    if evolve:
      self.evolve()

    return random.choice(moves)
  
  def solve_random(self, position: tuple[int, int] = None, tries: int = 100):
    if position is None:
      position = self.start

    mazes = [self]

    for itry in range(tries):
      path = [position]

      while not path[-1] is None:
        istep = len(path)

        if len(mazes) < istep:
          mazes.append(mazes[-1].clone())
          mazes[-1].evolve()

        cur_maze = mazes[istep - 1]
        cur_step = cur_maze.random_walk(path[-1])
        path.append(cur_step)

        if cur_step == self.finish:
          return True, path, itry+1
    
    return False, None, tries