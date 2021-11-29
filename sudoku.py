from numpy import flatnonzero
from numpy.lib.stride_tricks import as_strided


def solve_sudoku(board):
    """
    Solve a Sudoku board. The board is represented as a 9x9 array of values 0-9. The value 0
    represents a blank cell. The given board is modified so copy it if the original should be
    kept. 
    """
    working = init_working(board)

    nothing_found = 0 #  Number of times in a row we have failed to find anything
    while working.sum() != 0:  # while there are still things to do
        for axis in range(4):
            # Find the simple ones where only one possibility exists in a row/column/cell/block
            if not ((axis != 3 and find_simple_values(board, working, axis)) or
                    (axis == 3 and find_simple_values_in_block(board, working))):
                nothing_found += 1
                if nothing_found == 4:
                    # No simple ones found in cols, rows, cells, or blocks -> try harder
                    if not reduce_working_space(working): return board
                    nothing_found = 0
                    break
    return board


##### Utility Functions #####
def print_board(board):
    """Prints out a 9x9 board. 0 is printed as a space."""
    line = ('-'*13)
    for i in range(9):
        if i%3 == 0: print(line)
        for j in range(9):
            if j%3 == 0: print('|', end='')
            print(board[i,j] or ' ', end='')
        print('|')
    print(line)


def check_board(board, orig_board=None):
    """
    Checks the validity of a board (1-9 in each row, column, and 3x3 box).
    Optionally checks that no values in the original board (except 0) have changed.
    """
    if orig_board is not None:
        orig_board_filled = orig_board != 0
        if (board[orig_board_filled] != orig_board[orig_board_filled]).any():
            return False
    most_set = set(range(1,10))
    if (any(set(board[i,:]) != most_set for i in range(9)) or
        any(set(board[:,j]) != most_set for j in range(9)) or
        any(any(set(board[3*i:3*i+3,3*j:3*j+3].ravel()) != most_set for j in range(3)) for i in range(3))):
            return False
    return True


##### Utility Functions #####
def init_working(board):
    """
    Initialize and return the working array from a board. A working array is an 9x9x10 array of
    booleans where all the possible values are True and all others are False. In the initial case,
    the entries in the board that are 0 have all possibilities marked while the other values in the
    board have only 1 entry set.
    """
    from numpy import zeros
    working = zeros((9,9,10), dtype=bool)
    zero_set = set((0,))
    most_set = set(range(1,10))
    rows = [set(board[i,:])-zero_set for i in range(9)]
    cols = [set(board[:,j])-zero_set for j in range(9)]
    blocks = [[set(board[3*x:3*x+3,3*y:3*y+3].ravel())-zero_set for y in range(3)] for x in range(3)]
    for i,j in zip(*(board==0).nonzero()):
        w = most_set.difference(rows[i], cols[j], blocks[i//3][j//3])
        working[i,j,tuple(w)] = True
    return working


def set_board_value(board, working, i, j, n):
    """
    Update the board and working array by setting the value at position i, j on the board to n.
    """
    #print('Setting (%d,%d) to %d'%(i,j,n))
    board[i,j] = n
    working[i,:,n] = False # clear n from the row
    working[:,j,n] = False # clear n from the column
    working[i,j,:] = False # clear all values from a single cell
    x,y = (i//3)*3, (j//3)*3
    working[x:x+3,y:y+3,n] = False # clear n from the block


def find_simple_values(board, working, axis):
    """
    Find simple spots on the board/working to fill in values, depending on the axis:
     * axis=0 finds the simple values along cols
     * axis=1 finds the simple values along rows
     * axis=2 finds the simple values within a single cell
    """
    found = (working.sum(axis) == 1).nonzero()
    if len(found[0]) == 0: return False
    for a, b in zip(*found):
        if   axis == 0: i, j, n = flatnonzero(working[:,a,b])[0], a, b
        elif axis == 1: i, j, n = a, flatnonzero(working[a,:,b])[0], b
        elif axis == 2: i, j, n = a, b, flatnonzero(working[a,b,:])[0]
        set_board_value(board, working, i, j, n)
    return True


def find_simple_values_in_block(board, working):
    """Find simple spots on the board/working to fill in values within a block."""
    # views the working array as blocks
    working_blocks = as_strided(working, (3, 3, 3, 3, 10), (270, 30, 90, 10, 1)) # -> [x,y,xi,xj,n]
    found = (working_blocks.sum(2).sum(2) == 1).nonzero() # sum the entire block
    if len(found[0]) == 0: return False
    for x,y,n in zip(*found):
        for i, j in zip(*working_blocks[x,y,:,:,n].nonzero()):
            set_board_value(board, working, i+x*3,j+y*3,n)
    return True


##### Methods for finding non-simple values #####
# Note: these may not find any solutions on their own, but reduce the working space directly
def reduce_working_space(working):
    """
    Reduces the number of Trues in the working-space. Uses multiple methods and runs them until
    they no longer change anything. Returns False if no progress was possible.
    """
    methods = [__wr_method_1, __wr_method_2, __wr_method_3]
    looped = False
    rem = working.sum()
    while rem:
        old_rem = rem

        # Attempt methods 1-3
        for method in methods:
            method(working)
        rem = working.sum()

        if rem == old_rem:
            # No method helped this time around so far
            if not looped:
                # We are in trouble, we did nothing so far this entire call
                # Lets try some really 'expensive' methods
                for N in range(3, working.sum(2).max()+1):
                    __wr_method_4(working, N)
                    rem = working.sum()
                    if rem != old_rem:
                        # Yeah! It worked.
                        looped = True
                        break
                if rem != old_rem: continue
            break  # No method reduced, stopping
        looped = True
    return looped


def __wr_method_1(working):
    """
    This method looks for rows/columns where all of one value lies in one block. The rest of the
    blocks cannot have any of that value. Also assumes all standard methods have been tried.
    """
    working_block_rows = as_strided(working, (3,3,9,10),  (270,90,10,1)).any(1) # -> [x,j,n]
    found = (working_block_rows.sum(0) == 1).nonzero()
    for j,n in zip(*found):
        x,y = flatnonzero(working_block_rows[:,j,n])[0]*3, j//3*3
        y = tuple(y for y in range(y,y+3) if y != j)
        working[x:x+3,y,n] = False
    working_block_cols = as_strided(working, (9,3,3,10),  ( 90,30,10,1)).any(2) # -> [i,y,n]
    found = (working_block_cols.sum(1) == 1).nonzero()
    for i,n in zip(*found):
        x,y = i//3*3, flatnonzero(working_block_cols[i,:,n])[0]*3
        x = tuple(x for x in range(x,x+3) if x != i)
        working[x,y:y+3,n] = False


def __wr_method_2(working):
    """
    This method looks for blocks where all of one value lies in a single row/column. The rest of
    the row/column cannot have that value. Basically the "inverse" of method 1. Also assumes
    all standard methods have been tried.
    """
    working_blocks = as_strided(working, (3, 3, 3, 3, 10), (270, 30, 90, 10, 1)) # -> [x,y,xi,xj,n]
    found = ((working_blocks.sum(3) != 0).sum(2) == 1).nonzero()
    for x,y,n in zip(*found):
        f = working_blocks[x,y,:,:,n].nonzero()[0]
        i,j = f[0]+3*x, tuple(j for j in range(9) if j//3 != y)
        working[i,j,n] = False
    found = ((working_blocks.sum(2) != 0).sum(2) == 1).nonzero()
    for x,y,n in zip(*found):
        f = working_blocks[x,y,:,:,n].nonzero()[1]
        i,j = tuple(i for i in range(9) if i//3 != x), f[0]+3*y
        working[i,j,n] = False


def __wr_method_3(working):
    """
    This method looks for cells that have 2 values and are in a row/column/block with another
    cell that has the same 2 values. One of those two cells must contain the value, so the rest
    of the row/column/block cannot have them as possibilities.
    """
    found = (working.sum(2) == 2).nonzero()
    pairs = {}
    for i,j in zip(*found):
        vals = flatnonzero(working[i,j,:])
        if len(vals) != 2:
            # The data has changed significantly, restart
            __wr_method_3(working)
            return
        n,m = flatnonzero(working[i,j,:])
        p = pairs.setdefault((n,m), [])
        for (i2,j2) in p:
            x,y = i//3, j//3
            same_block = x == i2//3 and y == j2//3
            if same_block: # they lie in the same block (they can also lie in the same row/column as well)
                x *= 3; y *= 3
                working[x:x+3,y:y+3,(n,m)] = False
            if   i == i2: working[i,:,(n,m)] = False # they lie in the same row
            elif j == j2: working[:,j,(n,m)] = False # they lie in the same column
            elif not same_block: continue # just a coincidence
            # Add back the values to the cells themselves
            working[i, j, (n,m)] = True
            working[i2,j2,(n,m)] = True
        p.append((i,j))


def __wr_method_4(working, N=3):
    """
    This looks for groups of size N cells (default 3) in a single row/column/block that have <=N
    values each and together have exactly N unique values. Those values are removed from all
    other cells in the row/column/block. This is essentially the generalized version of method 3.
    This method is really expensive. It should only be used if all other methods failed.
    NOTE: currently only does row/column checking, not blocks
    """
    from itertools import combinations
    from numpy import unique, meshgrid
    data = working.sum(2)
    data = (data>0) & (data<=N)
    for i in range(9): # rows
        row = data[i,:]
        cells = flatnonzero(row)
        if len(cells) < N: continue
        for c in combinations(cells, N):
            vals = unique(working[i,c].nonzero()[1])
            if len(vals) == N:
                j = tuple(j for j in range(9) if j not in c)
                working[tuple(meshgrid(i,j,vals))] = False
    for j in range(9): # columns
        col = data[:,j]
        cells = flatnonzero(col)
        if len(cells) < N: continue
        for c in combinations(cells, N):
            vals = unique(working[c,j].nonzero()[1])
            if len(vals) == N:
                i = tuple(i for i in range(9) if i not in c)
                working[tuple(meshgrid(i,j,vals))] = False
    # TODO: This does seem to mostly work, but it is SLOW and causes odd side effects that should never
    # be able to happen (like the standard method stopping with 2 cells to go and giving up). I hope it
    # is never needed, otherwise some more debugging might be necessary
    #working_blocks = as_strided(working, (3, 3, 3, 3, 10), (270, 30, 90, 10, 1)) # -> [x,y,xi,xj,n]
    #data = working_blocks.sum(4)
    #data = (data>0) & (data<=N)
    #for x,y in product(xrange(3), xrange(3)): # blocks
    #    block = data[x,y,:,:]
    #    cells = flatnonzero(block)
    #    if len(cells) < N: continue
    #    for c in combinations(cells, N):
    #        ind = unravel_index(c, (3,3))
    #        vals = unique(working_blocks[x,y,ind[0],ind[1]].nonzero()[1])
    #        if len(vals) == N:
    #            xi,xj = unravel_index(tuple(i for i in xrange(9) if i not in c), (3,3))
    #            working[x,y,meshgrid(xi,xj,vals)] = False
# TODO: other ideas for methods:
#  * find two cells in a row/column/block that share two values that show up no where else in the row/column/block
#  * like above but generalized to N (for N>=3)

