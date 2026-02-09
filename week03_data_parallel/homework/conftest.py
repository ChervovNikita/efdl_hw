def pytest_addoption(parser):
    parser.addoption("--workers", action="store", type=int, default=2)
    parser.addoption("--hid-dim", action="store", type=int, default=16)
    parser.addoption("--batch-size", action="store", type=int, default=32)
