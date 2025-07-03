import numpy as np

def get_A_matrix_dwt_analysisFB_unit(h0,h1,N:int):
    """Returns Analysis Transform Matrix 'A'

    TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
    Copyright (C) 2025 Kishore Kumar Tarafdar

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""
    
    L = len(h0)
    __H_branch_row = lambda h0,N: np.concatenate((h0, np.zeros(int(N - len(h0)))))
    __H_start_row = lambda _row: np.roll(_row, shift=-L+2, axis=None)
    __H_start_row(__H_branch_row(h0,N))
    __H_branch= lambda _row: [np.roll(_row, shift=k*2, axis=None) for k in range(_row.shape[0]//2)]#-len(h0)+1)]
    H0 = __H_branch(__H_start_row(__H_branch_row(h0,N)))
    H1 = __H_branch(__H_start_row(__H_branch_row(h1,N)))
    # return np.concatenate((H0[:int(len(H0)/2)], H1[:int(len(H1)/2)]))
    return np.concatenate((H0, H1))

# A = get_A_matrix_dwt_analysisFB_unit(h0,h1,N)
# A.shape, A