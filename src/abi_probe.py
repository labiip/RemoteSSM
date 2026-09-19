import torch
print("ABI", torch._C._GLIBCXX_USE_CXX11_ABI)
print("cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE")
