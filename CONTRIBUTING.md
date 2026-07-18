# Contributing to Autonomous RC-Car

Thank you for contributing! To maintain the high quality, modularity, and clean separation of concerns in this project, please adhere to the following guidelines:

## Design Principles

1. **Contract-First (API-First)**:
   * Define shared types, enums, and interfaces in `src/common/` before implementing logic.
   * Implementation classes must inherit from and conform strictly to the abstract interfaces.

2. **Decoupled Architecture**:
   * **Core Logic Boundary**: Do not import heavy external libraries (like `cv2` or hardware serial communication) inside `src/common/` or `src/drive/`. Keep the core kinematics and controllers in pure Python using the `math` and `random` packages.
   * **Observers**: Loggers, visualizers, and other diagnostic tools must not send control signals. They should implement the observer pattern, subscribing to telemetry events emitted by the controller.

## Code Styling Guidelines

* **Line Length**: Keep lines under 100 characters.
* **Imports**: Always use clean, absolute imports relative to the `src/` directory (e.g. `from src.common.types import CarTelemetry`).
* **Header Comments**: Every code file must start with a clean, clear, and simple comment block at the very top explaining the file's purpose.
* **No Placeholders**: Do not check in incomplete work with TODO blocks in place of functional contracts; use basic `pass` signatures.

## Running Tests

All pull requests must pass the offline test suite before merging:
```bash
python3 -m pytest tests/
```
