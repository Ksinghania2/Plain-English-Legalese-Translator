import io
import sys
from legalese.cli import main


def test_cli_stdin(capsys):
    old_stdin = sys.stdin
    sys.stdin = io.StringIO("The Company shall indemnify the Customer.")
    try:
        rc = main(["translate", "-"])
    finally:
        sys.stdin = old_stdin
    assert rc == 0
    captured = capsys.readouterr()
    assert "must" in captured.out.lower() or "indemnif" in captured.out.lower()


def test_cli_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("The Tenant shall not assign this Agreement.", encoding="utf-8")
    rc = main(["translate", str(p), "--stats"])
    assert rc == 0


def test_cli_json(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("The Company shall indemnify the Customer.", encoding="utf-8")
    rc = main(["translate", str(p), "--json"])
    assert rc == 0
