# Goku-ELG Documentation

This directory contains the Sphinx documentation for Goku-ELG.

## Building the Documentation Locally

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `build/html/`. Open `build/html/index.html` in your browser.

### Build PDF Documentation

```bash
make latexpdf
```

The PDF will be in `build/latex/`.

### Clean Build Files

```bash
make clean
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main page
│   ├── introduction.rst     # Introduction
│   ├── installation.rst     # Installation guide
│   ├── quickstart.rst       # Quick start guide
│   ├── citation.rst         # Citation information
│   ├── license.rst          # License
│   ├── api/                 # API documentation
│   │   ├── index.rst
│   │   ├── gal_goku.rst
│   │   └── gal_goku_sims.rst
│   ├── tutorials/           # Tutorial pages
│   │   ├── index.rst
│   │   ├── basic_usage.rst
│   │   ├── hmf_emulation.rst
│   │   ├── galaxy_clustering.rst
│   │   └── advanced_topics.rst
│   ├── _static/             # Static files (images, CSS)
│   └── _templates/          # Custom templates
├── Makefile                 # Build commands (Unix)
├── make.bat                 # Build commands (Windows)
└── requirements.txt         # Documentation dependencies
```

## ReadTheDocs Integration

This documentation is configured to build on ReadTheDocs. The configuration is in `.readthedocs.yaml` at the repository root.

To set up on ReadTheDocs:

1. Go to https://readthedocs.org/
2. Import your GitHub repository
3. The build will automatically use `.readthedocs.yaml`
4. Documentation will be built on every commit

## Updating the Documentation

### Adding a New Page

1. Create a new `.rst` file in the appropriate directory
2. Add it to the relevant `toctree` directive in the parent `index.rst`
3. Rebuild the documentation

### Adding API Documentation

The API documentation is automatically generated from docstrings using Sphinx autodoc. To add documentation for a new module:

1. Add docstrings to your Python code (use NumPy or Google style)
2. Add the module to the appropriate API documentation file
3. Rebuild

### Adding Examples

Example code should be in code blocks:

```rst
.. code-block:: python

   import gal_goku
   # Your example code here
```

## Docstring Style

We use NumPy-style docstrings. Example:

```python
def my_function(param1, param2):
    """
    Brief description.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    
    Returns
    -------
    type
        Description of return value.
    
    Examples
    --------
    >>> my_function(1, 2)
    3
    """
    return param1 + param2
```

## Themes and Styling

The documentation uses the `sphinx_rtd_theme` (Read the Docs theme). To customize:

1. Edit `source/conf.py` for theme options
2. Add custom CSS in `source/_static/`
3. Add custom templates in `source/_templates/`

## Troubleshooting

### Build Errors

If you encounter build errors:

1. Check that all dependencies are installed
2. Run `make clean` before rebuilding
3. Check for syntax errors in `.rst` files
4. Verify import paths in `conf.py`

### Import Errors

If autodoc can't import modules:

1. Ensure packages are installed: `pip install -e src/gal_goku`
2. Check `sys.path` configuration in `conf.py`
3. Verify module structure and `__init__.py` files

### Missing References

If you see warnings about missing references:

1. Check that all referenced files exist
2. Verify `toctree` directives include all pages
3. Ensure cross-references use correct syntax

## Contributing

When contributing to the documentation:

1. Follow the existing structure and style
2. Test your changes locally before committing
3. Keep line length reasonable in `.rst` files
4. Use proper reStructuredText syntax
5. Include examples where appropriate

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Read the Docs](https://docs.readthedocs.io/)
