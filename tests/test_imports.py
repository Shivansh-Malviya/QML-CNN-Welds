"""Test that the qml_cnn_welds package can be imported."""

def test_import() -> None:
    """Ensure that the package can be imported when the src directory is added to sys.path."""
    import sys
    import pathlib

    # Append the src directory to the Python path
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.append(str(root / "src"))
    import qml_cnn_welds  # noqa: F401