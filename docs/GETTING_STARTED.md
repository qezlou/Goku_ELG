# Getting Started with Documentation

This guide will help you build and view the Goku-ELG documentation locally.

## Quick Start

### On Linux/Mac:

```bash
cd docs
./build_docs.sh
```

### On Windows:

```cmd
cd docs
build_docs.bat
```

## Manual Build Steps

If you prefer to build manually:

### 1. Install Dependencies

```bash
pip install -r docs/requirements.txt
```

### 2. Build Documentation

```bash
cd docs
make html
```

### 3. View Documentation

Open `docs/build/html/index.html` in your web browser, or run a local server:

```bash
cd docs/build/html
python -m http.server 8000
```

Then visit http://localhost:8000 in your browser.

## Building PDF Documentation

Requires LaTeX installation:

```bash
cd docs
make latexpdf
```

The PDF will be in `docs/build/latex/goku-elg.pdf`.

## Troubleshooting

### Import Errors

If Sphinx can't import the packages:

```bash
# Install packages in development mode
cd src/gal_goku
pip install -e .

cd ../gal_goku_sims
pip install -e .
```

### Missing Dependencies

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

### Clean Build

If you encounter issues:

```bash
cd docs
make clean
make html
```

## Development Workflow

When updating documentation:

1. Edit `.rst` files in `docs/source/`
2. Rebuild: `make html`
3. Refresh browser to see changes
4. Commit changes to git

## Auto-rebuild on Changes

For automatic rebuilding during development:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild source build/html
```

This will start a server at http://127.0.0.1:8000 that automatically rebuilds when files change.

## ReadTheDocs

The documentation is configured to build automatically on ReadTheDocs when you:

1. Push to the main branch
2. Create a pull request
3. Tag a release

Configuration is in `.readthedocs.yaml` at the repository root.

## Contributing

When contributing to documentation:

1. Follow reStructuredText conventions
2. Test builds locally before committing
3. Keep line length reasonable (80-100 characters)
4. Use code blocks for examples
5. Add docstrings to Python code

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── introduction.rst     # Introduction and overview
│   ├── installation.rst     # Installation instructions
│   ├── quickstart.rst       # Quick start guide
│   ├── quick_reference.rst  # Quick reference
│   ├── citation.rst         # How to cite
│   ├── license.rst          # License information
│   ├── api/                 # API documentation
│   └── tutorials/           # Tutorial pages
├── build/                   # Generated documentation (gitignored)
├── requirements.txt         # Documentation dependencies
├── Makefile                 # Build commands (Unix)
├── make.bat                 # Build commands (Windows)
├── build_docs.sh            # Automated build script (Unix)
└── build_docs.bat           # Automated build script (Windows)
```

## Need Help?

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- See [reStructuredText primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- Open an issue on GitHub
