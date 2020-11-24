> Copyright 2020 Nokia
> 
> Licensed under the BSD 3-Clause License
> 
> SPDX-License-Identifier: BSD-3-Clause

# TEST

This folder holds the basic testing framework for ZigZag.

## How to run the tests

```bash
# From the root of the git repository.
pytest
# Same, but also seing the output of ZigZag in real time.
pytest -s
# To run a single test file with test_base for instance.
pytest test/test_base.py
```

## How to add a test

To create a new simple test :arrow_down_small:

1. Create a new folder in [test/files](test/files) and add inside the input settings that can be used to reproduce the bug.
2. Make sure that you set the result directory of ZigZag to [test/results](test/results).
3. Copy the file called [test\_base](test/test_base.py) to the same folder, and give it the name of your test. :warning: The name of the file should always start with __test\___
4. Change the `directory` variable in the `test_main` function to be the directory where your input settings are. You may also need to change the name of your files if you used something other than the defaults.
5. In the _docstring_ of `test_main`, describe the behavior of the bug (ZigZag crashes or runs indefinitely for instance).
6. Return to the root folder and run `pytest $path_of_your_test` to see if the test works as intended.

You can then push your test to the appropriate branch and report it.

## Document Metadata

:date: File creation : Thu Oct  8 13:07:40 2020

:page_facing_up: Language : Markdown Documentation

## End of Document
