# Resources

- ReadMe.md - Always read this file into context.
- Lexicon.md - When writing new code always reference the lexicon first for semantics.
- Performance.md - When making significant changes to the codebase run the test suite and update the performance analysis.
- Layout.md - Use this file to help navigate the codebase to find relevant files. When files are moved or new files created - update the layout.
- notebooks/ - This directory contains all of the 'tutorial' notebooks for this API. The user experience of working with the API should feel like pandas/AnnData. The API should be simple to work with, but highly configurable and flexible. Reference these notebooks to understand what the user experience of working with the API feels like. Is it too complicated? Or simple and straightforwards...
- benchmarks/ - This directory contains the benchmark suite for comparing RadiObject vs MONAI vs TorchIO performance across storage backends.

For S3 access use the locally stored souzy-s3 credentials.