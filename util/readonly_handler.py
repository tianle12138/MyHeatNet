import os


def readonly_handler(func, path, execinfo):
    os.chmod(path, os.stat.S_IWRITE)
    func(path)
