import random

import matplotlib.pyplot as plt
import numpy as np

from .cell import Cell

class Maze():
  def __init__(self, matrix = None, shape: tuple[int, int] = None, start: tuple[int, int] = None, finish: tuple[int, int] = None, greens: list[tuple[int, int]] = []):
    if isinstance(matrix, np.ndarray) and len(matrix.shape) == 2:
      self.maze = matrix
      self.shape = matrix.shape
      self.rows = matrix.shape[0]
      self.cols = matrix.shape[1]

      # find start/ending points (also check input)
      self.start = None
      self.finish = None

      for pos, cell_type in np.ndenumerate(matrix):
        if cell_type == Cell.START:
          if self.start is None:
            self.start = pos
          else:
            raise RuntimeError('more than one starting point found')
        elif cell_type == Cell.FINISH:
          if self.finish is None:
            self.finish = pos
          else:
            raise RuntimeError('more than one ending point found')
        elif cell_type != Cell.WHITE and cell_type != Cell.GREEN:
          raise RuntimeError(f'invalid cell type {cell_type} at position {pos}')
        
      if self.start is None or self.finish is None:
        raise RuntimeError('input matrix is incomplete: missing start or finish tiles')

    elif not shape is None:
      self.shape = shape
      self.rows = shape[0]
      self.cols = shape[1]

      self.start = (0, 0) if start is None else start
      self.finish = (shape[0]-1, shape[1]-1) if finish is None else finish

      self.maze = np.full(shape, Cell.WHITE, dtype = int)
      self.maze[self.start] = Cell.START
      self.maze[self.finish] = Cell.FINISH
      for ij in greens:
        self.maze[ij] = Cell.GREEN

    else:
      raise RuntimeError('can not construct maze :(')
    
  def clone(self):
    return Maze(self.maze)

  def get_cell_indices(self, cell_type: Cell):
    return [pos for pos, c in np.ndenumerate(self.maze) if c == cell_type]

  def draw(self, position: tuple[int, int] = None, ms = 20):
    plt.matshow(self.maze, cmap = 'tab10')
    if not (position is None):
      plt.plot(position[1], position[0], 'kD', ms = ms)
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
      if self.maze[ij] == Cell.GREEN:
        count += 1
    
    return count
  
  def get_mutation(self, position: tuple[int, int]):
    type = self.maze[position]

    if type == Cell.START or type == Cell.FINISH:
      return type
    
    ngreen = self.count_green_neighbours(position)
    if type == Cell.WHITE and (1 < ngreen and ngreen < 5):
      return Cell.GREEN
    elif type == Cell.GREEN and (3 < ngreen and ngreen < 6):
      return Cell.GREEN
    else:
      return Cell.WHITE
  
  def evolve(self, overwrite: bool = True):
    other = np.full(self.maze.shape, Cell.WHITE, dtype = int)

    for pos in np.ndindex(self.shape):
      other[pos] = self.get_mutation(pos)

    if overwrite:
      self.maze = other

    return other

  def get_valid_moves(self, position: tuple[int, int] = None):
    if position is None:
      position = self.start

    if self.maze[position] == Cell.GREEN:
      return []
    
    nb = self.get_neighbours(position, include_diag = False)
    valid_moves = [ij for ij in nb if self.get_mutation(ij) != Cell.GREEN]
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

  def solve(self, starting_position: tuple[int, int] = None, max_steps: int = 100, verbose = False):
    if starting_position is None:
      starting_position = self.start

    # in case the position is the end tile (just in case)
    if self.maze[starting_position] == Cell.FINISH:
      return True

    mz = self.clone()

    survival_paths = [[(starting_position)]]

    for i in range(1, max_steps):
      next_survival_paths = []
      ending_points = []
      for path in survival_paths:
        for move in mz.get_valid_moves(path[-1]):
          if move == self.finish:
            return path + [move]
          if not move in ending_points:
            next_survival_paths.append(path + [move])
            ending_points.append(move)
      survival_paths = next_survival_paths
      mz.evolve()

      if verbose:
        print(f'step: {i-1}, paths: {len(survival_paths)}')

    return None