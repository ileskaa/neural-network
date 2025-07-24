# Weekly Report 3

## Challenges

It wasn't difficult to write methods to save and read parameters to and from .npz files, but interestingly enough, writing unit tests for those turned out to be quite complicated. The tests for the 'save' method were easier since I had already used unittest.mock.patch for another test method, and I was able to use the same method here. But writing tests for the parameter-loading method was another story. My initial plan was to mock the open() method from Python's `builtins` module to simulate opening a parameter containing file. But I kept running into all kinds of issues. I wasn't able to make the data saving and loading methods play nicely together. I tried all kinds of shenanigans but my program kept throwing unexpected errors at my face. After lots of trial and error, I found out about Python's `tempfile` module. And that saved my day. This module allowed me to create temporary files and directories, which were perfect for my unit tests.

## Next Steps
