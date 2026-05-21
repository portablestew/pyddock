"""Bug condition exploration test for subst drive trusted path mismatch.

**Validates: Requirements 1.1, 1.2, 1.3**

This test demonstrates the fix for the bug where `_caller_is_trusted()` previously
returned False when `co_filename` used a resolved real path but trusted prefixes
used an unresolved subst/junction path (or differed only in casing).

After the fix:
- `_caller_is_trusted()` applies `_normcase(_realpath(co_filename))` before comparison
- `install_import_hook()` builds trusted prefixes with `normcase(realpath(...))`
- Both sides are now in canonical form, so subst drives and case mismatches are handled

These tests verify the fix works correctly by simulating the normalized forms.
"""

from __future__ import annotations

import os
import sys
from types import CodeType, FrameType
from unittest.mock import MagicMock, patch

import pytest

from pyddock._runtime import SNIPPET_FILENAME
from pyddock._runtime import _caller_is_trusted, _normcase, _realpath


def _make_frame_chain(filenames: list[str]) -> MagicMock:
    """Build a mock frame chain with the given co_filename values.

    Returns the first frame in the chain. Each frame links to the next
    via .f_back. The last frame's .f_back is None.

    Since `_caller_is_trusted()` calls `sys._getframe(1)`, we mock
    `sys._getframe` to return the first frame directly (simulating
    the caller's frame).
    """
    frames: list[MagicMock] = []
    for filename in filenames:
        frame = MagicMock(spec=FrameType)
        frame.f_code = MagicMock(spec=CodeType)
        frame.f_code.co_filename = filename
        frames.append(frame)

    # Link frames via f_back
    for i in range(len(frames) - 1):
        frames[i].f_back = frames[i + 1]
    frames[-1].f_back = None

    return frames[0]


class TestSubstDrivePathMismatch:
    """Verify fix: subst drive path resolution in _caller_is_trusted().

    Property 1: Bug Condition - Subst/Junction Path Mismatch Blocks Trusted Imports

    The fix ensures that `_caller_is_trusted()` applies `_normcase(_realpath(...))`
    to `co_filename` before comparison, and `install_import_hook()` builds trusted
    prefixes with the same normalization. This means subst drives, junctions, and
    case mismatches are all resolved before the `startswith()` check.
    """

    def test_subst_drive_mismatch_blocks_trusted_import(self) -> None:
        """Subst drive: co_filename uses subst path, trusted prefix uses resolved real path.

        The fix resolves co_filename via _realpath before comparison.
        On this system J: is a subst for C:\\Perforce, so:
        - co_filename "J:\\project\\src\\mod.py" resolves to "C:\\Perforce\\project\\src\\mod.py"
        - trusted prefix is built as normcase(realpath("J:\\project\\src")) = "c:\\perforce\\project\\src"

        The fixed _caller_is_trusted() applies normcase(realpath(co_filename)) which
        resolves the subst drive, making the startswith() check succeed.
        """
        # Frame chain: workspace module -> snippet
        # The workspace module's co_filename uses the subst drive path
        # (as Python may record it depending on how the module was loaded)
        frame_chain = _make_frame_chain([
            "J:\\project\\src\\mod.py",  # subst drive path
            SNIPPET_FILENAME,             # agent snippet
        ])

        # Trusted prefixes are built by the fixed install_import_hook() using
        # normcase(realpath(...)), which resolves the subst drive and lowercases.
        # On this system: normcase(realpath("J:\\project\\src")) = "c:\\perforce\\project\\src"
        trusted_prefixes = (_normcase(_realpath("J:\\project\\src")),)

        with patch.object(sys, "_getframe", return_value=frame_chain):
            result = _caller_is_trusted(trusted_prefixes)

        # The fix resolves co_filename "J:\\project\\src\\mod.py" via realpath+normcase
        # to "c:\\perforce\\project\\src\\mod.py", which startswith "c:\\perforce\\project\\src"
        assert result is True, (
            "Fix verification: _caller_is_trusted() should return True when both "
            "co_filename and trusted prefix resolve through subst drive to the same "
            "canonical path after normcase(realpath(...)) normalization."
        )

    def test_case_mismatch_blocks_trusted_import(self) -> None:
        """Case mismatch: trusted prefix lowercase vs co_filename mixed case.

        On Windows, paths are case-insensitive. The fix applies normcase() to
        both sides, so casing differences no longer cause false negatives.
        """
        # Frame chain: workspace module -> snippet
        # co_filename uses mixed case (as Python records it)
        frame_chain = _make_frame_chain([
            "C:\\Perforce\\project\\src\\mod.py",  # mixed case
            SNIPPET_FILENAME,                       # agent snippet
        ])

        # Trusted prefix uses normcase form (lowercase on Windows)
        # This is what the fixed install_import_hook() produces
        trusted_prefixes = (_normcase("C:\\Perforce\\project\\src"),)

        with patch.object(sys, "_getframe", return_value=frame_chain):
            result = _caller_is_trusted(trusted_prefixes)

        # normcase(realpath("C:\\Perforce\\project\\src\\mod.py")) = "c:\\perforce\\project\\src\\mod.py"
        # which startswith "c:\\perforce\\project\\src"
        assert result is True, (
            "Fix verification: _caller_is_trusted() should return True when "
            "co_filename casing differs from trusted prefix because normcase() "
            "normalizes both to the same case."
        )

    def test_normal_drive_matching_path_works(self) -> None:
        """Sanity check: when paths are on a normal drive, _caller_is_trusted() works.

        This test confirms the function works correctly with normalized prefixes
        on a standard (non-subst) drive.
        """
        # Frame chain: workspace module -> snippet
        frame_chain = _make_frame_chain([
            "C:\\project\\src\\mod.py",  # normal path
            SNIPPET_FILENAME,             # agent snippet
        ])

        # Trusted prefix in normcase form (as the fixed install_import_hook() produces)
        trusted_prefixes = (_normcase(_realpath("C:\\project\\src")),)

        with patch.object(sys, "_getframe", return_value=frame_chain):
            result = _caller_is_trusted(trusted_prefixes)

        # normcase(realpath("C:\\project\\src\\mod.py")) should startswith
        # normcase(realpath("C:\\project\\src"))
        assert result is True, (
            "Fix verification: _caller_is_trusted() should return True when "
            "co_filename is under a trusted prefix on a normal drive."
        )

    def test_realpath_resolves_subst_for_co_filename(self) -> None:
        """Verify that realpath resolves the subst drive in co_filename.

        This directly tests the key mechanism: when co_filename uses the subst
        drive letter (J:), realpath resolves it to the real path (C:\\Perforce\\...),
        and normcase lowercases it, matching the trusted prefix.
        """
        # On this system J: -> C:\Perforce, so:
        resolved = _normcase(_realpath("J:\\project\\src\\mod.py"))
        expected = _normcase(_realpath("C:\\Perforce\\project\\src\\mod.py"))

        # Both should resolve to the same canonical path
        assert resolved == expected, (
            f"realpath should resolve subst drive: got {resolved!r} vs {expected!r}"
        )
