def pytest_addoption(parser):
    #parser.addoption('--rank', action='store', help="Rank", type=int)
    parser.addoption('--size', action='store', help="Size", type=int)
    parser.addoption('--intra_rank', action='store', help="Intra rank", type=int)
