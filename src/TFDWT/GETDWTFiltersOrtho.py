
from TFDWT.dbFBimpulseResponse import FBimpulseResponses

# import pywt
class GETDWTFiltersOrtho:
    """Get h, g, h_rec, g_rec FIR filters (reverse order) of Perfect Reconstruction DWT filter bank.
       
       Supports: Orthogonal and biorthogonal wavelet families.

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
    def __init__(self, wavelet: str):
        self.jsonfilepath = 'dbFBimpulseResp.json'
        # dbFBimpulseResp = self.__loadjson()
        # w = dbFBimpulseResp[wavelet]
        # self.h0n, self.h1n = dbFBimpulseResp[wavelet][0][0], dbFBimpulseResp[wavelet][0][1]
        # self.g0n, self.g1n = dbFBimpulseResp[wavelet][1][0], dbFBimpulseResp[wavelet][1][1]
        

        w = FBimpulseResponses[wavelet]
        self.h0n, self.h1n = FBimpulseResponses[wavelet][0][0], FBimpulseResponses[wavelet][0][1]
        self.g0n, self.g1n = FBimpulseResponses[wavelet][1][0], FBimpulseResponses[wavelet][1][1]
        



        type(w)
        # self.h0n , self.h1n, self.g0n, self.g1n
        

    def analysis(self):
        """Return Analysis filters"""
        # print(f'Impulse response (Analysis DT filt.):\n {self.h0n}\n {self.h1n},\n')
        return self.h0n, self.h1n

    def synthesis(self):
        """Return Synthesis filters"""
        # print(f'Impulse response (Synthesis DT filt.):\n {self.g0n}\n {self.g1n},\n')
        return self.g0n, self.g1n


    def __loadjson(self):
        # Load data from JSON file
        try:
            with open(self.jsonfilepath, 'r') as file:
                dbFBimpulseResp = json.load(file)
                print("Data loaded successfully:", dbFBimpulseResp)
        except FileNotFoundError:
            print("File not found. No data loaded.")
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        return dbFBimpulseResp


if __name__ == '__main__':
    # import pywt
    mother_wavelet = 'db6'
    mother_wavelet = 'bior3.1'
    # mother_wavelet = 'haar'
    w = GETDWTFiltersOrtho(mother_wavelet)
    h0n, h1n = w.analysis()
    g0n, g1n = w.synthesis()


