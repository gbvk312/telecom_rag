# Contributing to telecom_rag

Thank you for your interest in contributing to telecom_rag! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [How to Submit Pull Requests](#how-to-submit-pull-requests)
- [Code Style Guidelines](#code-style-guidelines)
- [Reporting Issues](#reporting-issues)

## Development Environment Setup

1. **Prerequisites**: Ensure you have Python 3.10+ installed on your system.

2. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/telecom_rag.git
   cd telecom_rag
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify the setup** by running any existing tests:
   ```bash
   python -m pytest tests/ -v
   ```

## How to Submit Pull Requests

1. **Create a feature branch** from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them with clear, descriptive commit messages:
   ```bash
   git add .
   git commit -m "feat: add a brief description of your change"
   ```

3. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** against the `master` branch of this repository.

5. **PR Requirements**:
   - Provide a clear description of the changes and their purpose.
   - Reference any related issues (e.g., `Closes #123`).
   - Ensure all CI checks pass.
   - Keep PRs focused — one feature or fix per PR.

6. **Code Review**: A maintainer will review your PR. Please be responsive to feedback and make requested changes promptly.

## Code Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style.
- Use meaningful variable and function names.
- Add docstrings to all public modules, classes, and functions.
- Keep functions focused and concise.
- Write type hints where applicable.
- Maximum line length: 120 characters.
- Use `snake_case` for functions and variables, `PascalCase` for classes.

## Reporting Issues

When reporting issues, please include:

1. **A clear and descriptive title.**
2. **Steps to reproduce** the issue.
3. **Expected behavior** vs. **actual behavior**.
4. **Environment details**: OS, Python version, relevant package versions.
5. **Screenshots or logs**, if applicable.

Use the [GitHub Issues](https://github.com/gbvk312/telecom_rag/issues) page to report bugs or request features.

---

Thank you for contributing! 🎉
