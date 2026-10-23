# CLAUDE.md

This file provides guidance to AI assistants when working with code in this repository.

## Development Environment

The Python virtual environment is likely located at `./venv`, but this is just a guess - the user may have it elsewhere or may have already activated a virtualenv. If commands like `python` or `pytest` work directly, the environment is probably already active.

If you need to use the venv explicitly:

```bash
./venv/bin/python -m pytest src/toil/test/path/to/test.py -v
./venv/bin/python -c "import toil; print(toil.__version__)"
```

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies and Toil in editable mode
make prepare
make develop extras=[all]  # Install with all extras

# Verify installation
python -c "import toil; print(toil.__version__)"
```

## Running Tests

Tests use pytest. Example commands:

```bash
# Run a specific test file
./venv/bin/python -m pytest src/toil/test/server/safeFileTest.py -v

# Run a specific test
./venv/bin/python -m pytest src/toil/test/server/safeFileTest.py::TestSafeFileInterleaving::test_reader_blocked_while_writer_holds_lock -v

# Run tests with a keyword filter
./venv/bin/python -m pytest src/toil/test -k "safe" -v
```

## Running Make Targets (mypy, tests, etc.)

The `Makefile` targets require the virtualenv to be activated. Some targets (like `test_debug`) enforce this with a `check_venv` guard that checks for `VIRTUAL_ENV` in the environment. Set `PATH` and `VIRTUAL_ENV` to satisfy this without needing `source`:

```bash
PATH="./venv/bin:$PATH" VIRTUAL_ENV=./venv make mypy
PATH="./venv/bin:$PATH" VIRTUAL_ENV=./venv make test_debug tests='src/toil/test/path/to/test.py::TestClass::test_name'
```

### Key Make Targets

```bash
# Install development requirements
make prepare

# Install Toil in editable mode (with optional extras)
make develop extras=[cwl]  # CWL support
make develop extras=[wdl]  # WDL support  
make develop extras=[aws]  # AWS support
make develop extras=[all]  # All extras

# Run all tests (requires Docker for some tests)
make test

# Run tests without Docker/Internet requirements
make test_offline

# Run tests with live debugging output
make test_debug

# Run specific test or test file
make test tests='src/toil/test/server/safeFileTest.py::TestSafeFileInterleaving::test_reader_blocked_while_writer_holds_lock'

# Code formatting and linting
make format      # Black formatting
make mypy        # Type checking
make sort_imports # Organize imports

# Build distribution package
make dist

# Clean build artifacts
make clean
```

## Running Individual WDL Spec Unit Tests

The WDL spec embeds example workflows as unit tests (under the `wdl-1.1` and `wdl-1.2` branches of `https://github.com/openwdl/wdl`). `TestWDLConformance.test_single_unit_test` in `src/toil/test/wdl/wdltoil_test.py` runs one such test at a time and is controlled by environment variables:

- `WDL_UNIT_TEST_ID`: which test to run (default: `glob_task`)
- `WDL_UNIT_TEST_VERSION`: WDL version (default: `1.1`)

```bash
WDL_UNIT_TEST_ID=serde_pair ./venv/bin/python -m pytest \
  src/toil/test/wdl/wdltoil_test.py::TestWDLConformance::test_single_unit_test \
  --timeout=300 -v -s
```

This test clones remote git repos and may be slow. Many WDL spec tasks run inside containers, so **Docker must be running** — if the test fails with a Docker connection error, ask the user to start Docker before retrying.

## Architecture Overview

Toil is a scalable workflow management system with these key components:

### Core Architecture

- **Job Management**: [src/toil/job.py](src/toil/job.py) - Job definitions and execution logic
- **Leader Process**: [src/toil/leader.py](src/toil/leader.py) - Orchestrates workflow execution
- **Worker Processes**: [src/toil/worker.py](src/toil/worker.py) - Execute individual jobs
- **State Management**: [src/toil/toilState.py](src/toil/toilState.py) - Maintains execution state
- **Configuration**: [src/toil/common.py](src/toil/common.py) - Core configuration and CLI parsing

### Storage Systems

- **Job Stores**: [src/toil/jobStores/](src/toil/jobStores/) - Persistence backends (File, AWS, Azure, Google Cloud)
- **File Stores**: [src/toil/fileStores/](src/toil/fileStores/) - File management across distributed systems

### Execution Environments

- **Batch Systems**: [src/toil/batchSystems/](src/toil/batchSystems/) - Execution backends (SingleMachine, Slurm, Kubernetes, AWS Batch, etc.)
- **Provisioners**: [src/toil/provisioners/](src/toil/provisioners/) - Cloud resource provisioning

### Workflow Languages

- **CWL Support**: [src/toil/cwl/](src/toil/cwl/) - Common Workflow Language implementation
- **WDL Support**: [src/toil/wdl/](src/toil/wdl/) - Workflow Description Language implementation

### Service Infrastructure

- **Server**: [src/toil/server/](src/toil/server/) - REST API and web services
- **Message Bus**: [src/toil/bus.py](src/toil/bus.py) - Inter-process communication
- **Utilities**: [src/toil/utils/](src/toil/utils/) - Shared utility functions

### Shared Libraries

- **Lib**: [src/toil/lib/](src/toil/lib/) - Reusable components (io, memoize, retry, threading, etc.)
- **Options**: [src/toil/options/](src/toil/options/) - Command-line option definitions

## Code Style

Docstrings state the contract (what, not how). Implementation strategy goes in comments on the relevant code. Explain a concept once in a canonical docstring; reference it elsewhere. Names must be precise: no redundant qualifiers, invented terms, overclaiming, or test content as examples.

### Formatting and Linting

- Follow Black formatting (enforced via `make format`)
- Use isort for import sorting (`make sort_imports`)
- Type hints throughout the codebase
- Mypy type checking (`make mypy`)

## Development Patterns

### Testing Patterns

- Tests are organized under [src/toil/test/](src/toil/test/)
- Use pytest fixtures for test setup/teardown
- Many tests require Docker for container execution
- Cloud-based tests require appropriate credentials
- Environment variables control test behavior (`TOIL_SKIP_DOCKER`, `TOIL_SKIP_ONLINE`)

### Dependency Management

- Core dependencies in [requirements.txt](requirements.txt)
- Optional extras in requirements-*.txt files
- Extras include: aws, cwl, google, kubernetes, wdl, server, encryption, mesos, htcondor
- Platform-specific dependencies (htcondor not available on macOS)

## Key Environment Variables

```bash
# Docker configuration for appliance containers
TOIL_DOCKER_REGISTRY="quay.io/ucsc_cgl"  # Default registry
TOIL_DOCKER_NAME="toil"  # Base image name
TOIL_APPLIANCE_SELF=""  # Override full image specification

# Test execution controls
TOIL_SKIP_DOCKER=True  # Skip Docker-requiring tests
TOIL_SKIP_ONLINE=True  # Skip Internet-requiring tests
TOIL_OWNER_TAG="shared"  # Resource tagging for cloud tests

# Debug and development
TOIL_DEBUG=True  # Enable debug logging
TOIL_HISTORY=0  # Disable execution history tracking
TOIL_CHECK_ENV="True"  # Strict environment checking
```

## Troubleshooting

**Common Issues:**

- **Docker not running**: Required for many tests, start Docker daemon
- **Virtualenv not active**: Use `source venv/bin/activate` or prefix commands with `./venv/bin/`
- **Missing dependencies**: Run `make prepare` to install build requirements
- **Cloud credential issues**: Tests requiring AWS/Google Cloud may fail without proper credentials
- **HTCondor on macOS**: HTCondor extra is not available on macOS

**Performance Tips:**

- Use `make test_offline` for faster iteration without Docker/Internet requirements
- Set `threads="1"` make variable for single-threaded test execution when debugging
- Use `make test_debug` for live logging output during test failures
- Set `cov=""` to disable coverage reporting for faster test runs

## Extending Toil

When adding new features:

1. **Batch Systems**: Extend [AbstractBatchSystem](src/toil/batchSystems/abstractBatchSystem.py)
2. **Job Stores**: Extend [AbstractJobStore](src/toil/jobStores/abstractJobStore.py)
3. **File Stores**: Extend [AbstractFileStore](src/toil/fileStores/abstractFileStore.py)
4. **Workflow Languages**: Follow patterns in CWL/WDL implementations
5. **Tests**: Add comprehensive tests under the appropriate test directory

Remember that Toil supports both Python API workflows and standardized workflow languages (CWL/WDL), so consider both interfaces when extending functionality.

## Project Structure Quick Reference

- **Core**: `src/toil/{job,leader,worker,toilState,common}.py`
- **Storage**: `src/toil/{jobStores,fileStores}/`
- **Execution**: `src/toil/{batchSystems,provisioners}/`
- **Workflows**: `src/toil/{cwl,wdl}/`
- **Services**: `src/toil/{server,bus,utils}/`
- **Shared**: `src/toil/{lib,options}/`
- **Tests**: `src/toil/test/`

Use the Makefile for common development tasks and refer to the comprehensive documentation at <https://toil.readthedocs.io/> for detailed architecture and usage guidelines.
