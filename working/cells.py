import uuid


def new_cell(curr_id, position, ):
    idx = uuid.uuid4()
    _cell = Cell(idx)
    _cell.curr_id = curr_id
    _cell.position = position
    return _cell


class Cell(object):
    def __init__(self, idx):
        self._idx = idx
        self.positions = []
        self.past_ids = []
        self.costs = []

    @property
    def idx(self):
        return self._idx

    @property
    def cost(self):
        return self.costs

    @cost.setter
    def cost(self, value):
        self.cost.append(value)

    @property
    def curr_id(self):
        return self._curr_id

    @curr_id.setter
    def curr_id(self, value):
        self._curr_id = value
        self.past_ids.append(value)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self.positions.append(value)

    @property
    def cell_positions(self):
        return self.positions


class Cells(object):
    def __init__(self):
        self.cells = {}

    def add_cell(self, cell):
        assert (isinstance(cell, Cell))
        self.cells[cell._idx] = cell

    def modify_cell_id(self, cell, idx):
        self.cells[cell].curr_id = idx

    def get_cell(self, cell_id):
        return self.cells[cell_id]

    def get_cells(self):
        return self.cells

    def get_cell_ids(self):
        return [self.cells]

    def cellcount(self):
        return len(self.cells)
