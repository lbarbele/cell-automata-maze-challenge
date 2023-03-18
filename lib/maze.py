import random

import matplotlib.pyplot as plt
import numpy as np

from .cell import Cell

class Maze():
  """
  The Maze class represents a rectangular maze of cell automata and provides methods
  to analyze its dynamics.
  """

  def __init__(self, matrix: np.ndarray):
    # check input
    if not isinstance(matrix, np.ndarray):
      raise RuntimeError('input matrix is invalid (expected a numpy array)')

    if len(matrix.shape) != 2:
      raise RuntimeError(f'input matrix has {len(matrix.shape)} dimensions whereas 2 were expected')

    # copy the input matrix with integer dtype and retrieve its shape information
    self.maze = np.array(matrix, dtype = int)
    self.shape = self.maze.shape
    self.rows = self.shape[0]
    self.cols = self.shape[1]

    # loop over cells to find start/end and check values
    self.start = None
    self.finish = None

    for pos, cell in np.ndenumerate(matrix):
      if cell == Cell.START and self.start is None:
        self.start = pos
      elif cell == Cell.FINISH and self.finish is None:
        self.finish = pos
      elif cell == Cell.START and not (self.start is None):
        raise RuntimeError('more than one starting point found in matrix')
      elif cell == Cell.FINISH and not (self.finish is None):
        raise RuntimeError('more than one finishing point found in matrix')
      elif cell != Cell.WHITE and cell != Cell.GREEN:
        raise RuntimeError(f'invalid cell found (value = {cell})')

    # check if start and finish were found
    if self.start is None or self.finish is None:
      raise RuntimeError('input matrix is incomplete: missing start or finish tiles')

  def clone(self):
    return Maze(self.maze)
  
  def count_green_neighbours(self, position: tuple[int, int]):
    count = 0

    for ij in self.get_neighbours(position):
      if self.maze[ij] == Cell.GREEN:
        count += 1
    
    return count

  def draw(self, position: tuple[int, int] = None, ms = 20):
    plt.matshow(self.maze, cmap = 'tab10')
    if not (position is None):
      plt.plot(position[1], position[0], 'kD', ms = ms)
      moves = np.array(self.get_valid_moves(position))
      if len(moves) > 0:
        plt.plot(moves[:, 1], moves[:, 0], 'k+')
    plt.show()
  
  def evolve(self, overwrite: bool = True):
    other = np.full(self.maze.shape, Cell.WHITE, dtype = int)

    for pos in np.ndindex(self.shape):
      other[pos] = self.get_mutation(pos)

    if overwrite:
      self.maze = other

    return other

  def get_cell_indices(self, cell_type: Cell):
    return [pos for pos, c in np.ndenumerate(self.maze) if c == cell_type]
  
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

  def get_valid_moves(self, position: tuple[int, int] = None):
    if position is None:
      position = self.start

    if self.maze[position] == Cell.GREEN:
      return []
    
    nb = self.get_neighbours(position, include_diag = False)
    valid_moves = [ij for ij in nb if self.get_mutation(ij) != Cell.GREEN]
    return valid_moves
  
  def random_walk(self, position: tuple[int, int], evolve: bool = False):
    if position is None:
      return None
    
    moves = self.get_valid_moves(position)

    if len(moves) == 0:
      return None
    
    if evolve:
      self.evolve()

    return random.choice(moves)

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