
from TFDWT.dbFBimpulseResponse import FBimpulseResponses

# import pywt
class FetchAnalysisSynthesisFilters:
    """TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
    Copyright 2026 Kishore Kumar Tarafdar

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    
        Get h, g, h_rec, g_rec FIR filters (reverse order) of Perfect Reconstruction DWT filter bank.
    
    Supports: Orthogonal and biorthogonal wavelet families."""
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
    w = OrthogonalBiorthogonalFilters(mother_wavelet)
    h0n, h1n = w.analysis()
    g0n, g1n = w.synthesis()

