"""Command line argument parsing."""
from camzmq.cli import _size


def test_size_is_width_by_height_on_the_command_line_and_height_width_in_code():
    assert _size("320x240") == (240, 320)
    assert _size("102X64") == (64, 102)
