"""Tests for the rinexpy CLI."""

from __future__ import annotations

import pytest

from rinexpy.cli import build_parser, main

from .conftest import fixture


def test_build_parser_smoke():
    parser = build_parser()
    ns = parser.parse_args(["read", "x.10o"])
    assert ns.cmd == "read"
    assert ns.file == "x.10o"


def test_parser_times_subcommand():
    parser = build_parser()
    ns = parser.parse_args(["times", "x.10o"])
    assert ns.cmd == "times"


def test_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_times(capsys):
    fn = str(fixture("demo.10o"))
    rc = main(["times", fn])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 epochs" in out


def test_cli_info(capsys):
    fn = str(fixture("demo.10o"))
    rc = main(["info", fn])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RINEX VERSION" in out or "rinextype" in out


def test_cli_read(capsys):
    fn = str(fixture("minimal2.10o"))
    rc = main(["read", fn])
    assert rc == 0
    out = capsys.readouterr().out
    assert "xarray" in out.lower() or "<xarray" in out


def test_cli_convert(tmp_path):
    pytest.importorskip("netCDF4")
    src = fixture("demo.10o")
    out_dir = tmp_path
    rc = main(["convert", str(src.parent), src.name, "--out", str(out_dir)])
    assert rc == 0
    # Output file should exist
    assert (out_dir / f"{src.name}.nc").is_file()


def test_cli_unknown_file_returns_error():
    rc = main(["read", "/no/such/file.10o"])
    assert rc == 1
