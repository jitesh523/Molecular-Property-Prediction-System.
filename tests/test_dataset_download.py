"""The CI dataset selector must not trigger every remote dataset download."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    "download_molnet", Path(__file__).resolve().parents[1] / "scripts/download_molnet_datasets.py"
)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def test_downloader_limits_downloads_to_requested_dataset():
    with patch.object(download, "download_dataset", return_value=True) as fetch, \
         patch("sys.argv", ["download", "--datasets", "bbbp"]):
        download.main()
    fetch.assert_called_once_with("bbbp", download.DATASETS["bbbp"])


def test_downloader_reports_failure_to_ci():
    with patch.object(download, "download_dataset", return_value=False), \
         patch("sys.argv", ["download", "--datasets", "bbbp"]), \
         pytest.raises(SystemExit) as failure:
        download.main()
    assert failure.value.code == 1


def test_default_still_downloads_every_registered_dataset():
    fetch = Mock(return_value=True)
    with patch.object(download, "download_dataset", fetch), patch("sys.argv", ["download"]):
        download.main()
    assert [call.args[0] for call in fetch.call_args_list] == list(download.DATASETS)
