# Contributing to PADDLE

Thank you for your interest in contributing to PADDLE (Predictive Analytics and Data-driven Decision Learning Engine)!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/paddle.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install in development mode: `pip install -e .`
6. Install development dependencies: `pip install pytest pytest-cov black flake8`

## Development Workflow

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/`
4. Run linting: `black paddle/ tests/ examples/`
5. Commit your changes: `git commit -m "Description of changes"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting a PR
- Aim for high test coverage (>80%)
- Use pytest for testing framework

## Adding New Models

When adding new predictive or prescriptive models:

1. Add the model class to the appropriate module (`paddle/models/predictive.py` or `paddle/models/prescriptive.py`)
2. Create a data generator method in `paddle/data/generator.py`
3. Write comprehensive tests in `tests/`
4. Create an example script in `examples/`
5. Update the README with information about the new model

## Documentation

- Update README.md for major features
- Add docstrings following Google style
- Include usage examples in docstrings
- Update API documentation as needed

## Reporting Issues

- Use the GitHub issue tracker
- Provide a clear description of the issue
- Include steps to reproduce
- Specify your environment (OS, Python version, etc.)

## Questions?

Feel free to open an issue for questions or discussions about the project.

Thank you for contributing!
